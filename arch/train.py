"""Distributed retriever training for a pinned, non-legacy stack.

Target versions:
    sentence-transformers 5.1.x
    transformers          4.52.x
    torch                 2.6.x

Launch with torchrun so a saved Accelerate profile cannot silently enable
FSDP or DeepSpeed:

    torchrun --standalone --nproc_per_node=2 train_retriever_st51.py \
        --data data/train.jsonl \
        --corpus data/corpus.jsonl \
        --model intfloat/multilingual-e5-base \
        --output runs/me5-base \
        --cache-dir retriever-cache \
        --negatives 8 \
        --batch-size 256 \
        --mini-batch-size 32 \
        --push-to-hub \
        --hub-model-id your-name/me5-base-retriever

With --push-to-hub, the latest resumable checkpoint is uploaded to the
repository's last-checkpoint directory at every local checkpoint save. Set
HF_TOKEN in the environment or authenticate once with `hf auth login`; never
put a token directly in this command. TensorBoard logs are written to
OUTPUT/tensorboard by default.

Input files:
    corpus.jsonl: {"id": "doc-id", "text": "..."}
    train.jsonl:  {
        "query": "...",
        "positives": ["doc-id", ...],
        "negatives": ["doc-id" | {"id": "doc-id", ...}, ...],
        "cluster": "group-id"  # optional; source_case/query are fallbacks
    }

The training dataset stores queries and document IDs only. Corpus text lives in
a memory-mapped Arrow dataset and is fetched for the current batch. This avoids
expanding millions of samples into tens of gigabytes of duplicate Python text.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterator

# This script deliberately supports plain DDP only. These assignments happen
# before importing Transformers/Accelerate so external profiles cannot opt in to
# a different distributed backend.
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"

import torch
from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk
from filelock import FileLock
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rank=%(rank)s] %(levelname)s %(message)s",
)


class RankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = int(os.environ.get("RANK", "0"))
        return True


for handler in logging.getLogger().handlers:
    handler.addFilter(RankFilter())
log = logging.getLogger("retriever-training")

CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Version and argument validation
# ---------------------------------------------------------------------------

def require_minor(package: str, expected_major: int, expected_minor: int) -> None:
    try:
        installed = version(package)
    except PackageNotFoundError as error:
        raise RuntimeError(f"required package is not installed: {package}") from error

    numeric = installed.split("+", 1)[0].split(".")
    actual = tuple(int(part) for part in numeric[:2])
    expected = (expected_major, expected_minor)
    if actual != expected:
        raise RuntimeError(
            f"{package} {installed} is installed; this script requires "
            f"{expected_major}.{expected_minor}.x"
        )


def validate_versions() -> None:
    require_minor("sentence-transformers", 5, 1)
    require_minor("transformers", 4, 52)
    require_minor("torch", 2, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs; useful for a short smoke test",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Per GPU")
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--max-eval-samples", type=int, default=2_000)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=1_000)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument(
        "--report-to",
        choices=("tensorboard", "none"),
        default="tensorboard",
        help="Metrics backend; TensorBoard is enabled by default",
    )
    parser.add_argument(
        "--logging-dir",
        default=None,
        help="TensorBoard directory (default: OUTPUT/tensorboard)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run label shown in logging integrations",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload models and checkpoints to the Hugging Face Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Destination repository, e.g. username/model-name",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the Hub repository as private (ignored if it already exists)",
    )
    parser.add_argument(
        "--hub-strategy",
        choices=("end", "every_save", "checkpoint", "all_checkpoints"),
        default="checkpoint",
        help=(
            "checkpoint uploads the latest resumable checkpoint as "
            "last-checkpoint; all_checkpoints retains every remote checkpoint"
        ),
    )
    parser.add_argument(
        "--hub-always-push",
        action="store_true",
        help="Queue a new upload even if the previous asynchronous upload is unfinished",
    )

    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--document-prefix", default="passage: ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--gather-across-devices", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)

    parser.add_argument(
        "--lora",
        action="store_true",
        help="Train LoRA adapters instead of full fine-tuning",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    if args.negatives < 0:
        parser.error("--negatives must be non-negative")
    if args.batch_size < 1 or args.mini_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.mini_batch_size > args.batch_size:
        parser.error("--mini-batch-size cannot exceed --batch-size")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.save_steps % args.eval_steps != 0:
        parser.error("--save-steps must be a multiple of --eval-steps")
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub-model-id is required with --push-to-hub")
    if not args.push_to_hub and args.hub_model_id:
        parser.error("--hub-model-id requires --push-to-hub")
    return args


def validate_hub_authentication(args: argparse.Namespace) -> None:
    """Fail before costly data/model setup when Hub credentials are absent."""
    if not args.push_to_hub:
        return
    from huggingface_hub import get_token

    if get_token() is None:
        raise RuntimeError(
            "--push-to-hub requires authentication. Set HF_TOKEN to a write "
            "token or run `hf auth login` before launching torchrun."
        )


# ---------------------------------------------------------------------------
# Compact disk-backed dataset construction
# ---------------------------------------------------------------------------

def file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_key(args: argparse.Namespace) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "data": file_identity(Path(args.data)),
        "corpus": file_identity(Path(args.corpus)),
        "negatives": args.negatives,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(serialized, digest_size=12).hexdigest()


def is_validation_cluster(cluster: str, fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        f"{seed}\0{cluster}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64) < fraction


def load_valid_corpus_ids(path: Path) -> set[str]:
    valid: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (record.get("text") or "").strip():
                valid.add(str(record["id"]))
    return valid


def iter_reference_rows(
    data_path: str,
    corpus_path: str,
    negatives_per_query: int,
    validation_fraction: float,
    seed: int,
    split: str,
) -> Iterator[dict[str, str]]:
    """Yield only query text and document IDs; never duplicate corpus text."""
    valid_ids = load_valid_corpus_ids(Path(corpus_path))
    emitted = 0
    skipped = 0

    with Path(data_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            record = json.loads(line)
            query = (record.get("query") or "").strip()

            positives = [str(item) for item in record.get("positives", [])]
            positives = [item for item in positives if item in valid_ids]
            positive_ids = set(positives)

            raw_negatives = [
                item.get("id") if isinstance(item, dict) else item
                for item in record.get("negatives", [])
            ]
            negatives = []
            seen_negatives: set[str] = set()
            for item in raw_negatives:
                if item is None:
                    continue
                document_id = str(item)
                if (
                    document_id in valid_ids
                    and document_id not in positive_ids
                    and document_id not in seen_negatives
                ):
                    seen_negatives.add(document_id)
                    negatives.append(document_id)

            if not query or not positives or len(negatives) < negatives_per_query:
                skipped += 1
                continue

            cluster = str(
                record.get("cluster")
                or record.get("source_case")
                or query
            )
            row_is_validation = is_validation_cluster(
                cluster, validation_fraction, seed
            )
            if row_is_validation != (split == "validation"):
                continue

            row_seed = int.from_bytes(
                hashlib.blake2b(
                    f"{seed}\0{line_number}".encode("utf-8"), digest_size=8
                ).digest(),
                "big",
            )
            rng = random.Random(row_seed)
            row = {
                "anchor": query,
                "positive": rng.choice(positives),
            }
            for index, document_id in enumerate(
                rng.sample(negatives, negatives_per_query)
            ):
                row[f"negative_{index}"] = document_id
            emitted += 1
            yield row

    log.info("%s: emitted=%d skipped=%d", split, emitted, skipped)


def reference_features(negative_count: int) -> Features:
    columns = {
        "anchor": Value("string"),
        "positive": Value("string"),
    }
    columns.update(
        {f"negative_{index}": Value("string") for index in range(negative_count)}
    )
    return Features(columns)


def build_reference_cache(args: argparse.Namespace, destination: Path) -> None:
    """Create one atomic cache shared by every torchrun worker."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if (destination / "dataset_dict.json").exists():
            return

        temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        generator_cache = temporary / "generator"
        saved = temporary / "dataset"
        generator_cache.mkdir(parents=True)

        common = {
            "data_path": str(Path(args.data).resolve()),
            "corpus_path": str(Path(args.corpus).resolve()),
            "negatives_per_query": args.negatives,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        }
        features = reference_features(args.negatives)
        log.info("building compact dataset cache at %s", destination)

        train = Dataset.from_generator(
            iter_reference_rows,
            gen_kwargs={**common, "split": "train"},
            features=features,
            cache_dir=str(generator_cache),
            keep_in_memory=False,
            writer_batch_size=1_000,
        )
        validation = Dataset.from_generator(
            iter_reference_rows,
            gen_kwargs={**common, "split": "validation"},
            features=features,
            cache_dir=str(generator_cache),
            keep_in_memory=False,
            writer_batch_size=1_000,
        )
        DatasetDict({"train": train, "validation": validation}).save_to_disk(
            str(saved), max_shard_size="1GB"
        )

        del train, validation
        gc.collect()
        os.replace(saved, destination)
        shutil.rmtree(temporary)
        log.info("published dataset cache at %s", destination)


def load_corpus(path: Path, cache_dir: Path) -> Dataset:
    corpus = load_dataset(
        "json",
        data_files={"train": str(path.resolve())},
        split="train",
        cache_dir=str(cache_dir),
        keep_in_memory=False,
    )
    missing = {"id", "text"}.difference(corpus.column_names)
    if missing:
        raise ValueError(f"corpus is missing columns: {sorted(missing)}")
    return corpus.select_columns(["id", "text"])


def index_corpus(corpus: Dataset) -> dict[str, int]:
    index: dict[str, int] = {}
    offset = 0
    for batch in corpus.iter(batch_size=10_000):
        for relative, document_id in enumerate(batch["id"]):
            index[str(document_id)] = offset + relative
        offset += len(batch["id"])
    log.info("indexed %d corpus documents", len(index))
    return index


def attach_text_lookup(
    references: Dataset,
    corpus: Dataset,
    corpus_index: dict[str, int],
    query_prefix: str,
    document_prefix: str,
) -> Dataset:
    document_columns = [
        column for column in references.column_names if column != "anchor"
    ]

    def transform(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        batch_size = len(batch["anchor"])
        output = {
            "anchor": [query_prefix + query for query in batch["anchor"]]
        }

        row_indices: list[int] = []
        for column in document_columns:
            try:
                row_indices.extend(corpus_index[item] for item in batch[column])
            except KeyError as error:
                raise KeyError(
                    f"document ID missing from corpus: {error.args[0]}"
                ) from error

        texts = corpus[row_indices]["text"]
        for column_number, column in enumerate(document_columns):
            start = column_number * batch_size
            output[column] = [
                document_prefix + (text or "").strip()
                for text in texts[start : start + batch_size]
            ]
        return output

    return references.with_transform(transform)


def prepare_datasets(
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, Dataset, dict[str, int]]:
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    references_path = root / f"references-{cache_key(args)}"
    build_reference_cache(args, references_path)

    references = load_from_disk(str(references_path), keep_in_memory=False)
    corpus = load_corpus(Path(args.corpus), root / "corpus-arrow")
    corpus_index = index_corpus(corpus)

    train = attach_text_lookup(
        references["train"],
        corpus,
        corpus_index,
        args.query_prefix,
        args.document_prefix,
    )
    validation_refs = references["validation"]
    if 0 < args.max_eval_samples < len(validation_refs):
        validation_refs = validation_refs.select(range(args.max_eval_samples))
    validation = attach_text_lookup(
        validation_refs,
        corpus,
        corpus_index,
        args.query_prefix,
        args.document_prefix,
    )
    log.info(
        "dataset ready: train=%d validation=%d corpus=%d",
        len(train),
        len(validation),
        len(corpus),
    )
    return train, validation, corpus, corpus_index


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------

def freeze_unused_pooler(model: SentenceTransformer) -> None:
    """Freeze HF's classification pooler; ST uses its own pooling module."""
    first_module = model._first_module()
    backbone = getattr(first_module, "auto_model", None)
    pooler = getattr(backbone, "pooler", None)
    if pooler is None:
        return

    trainable = [parameter for parameter in pooler.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.requires_grad_(False)
    if trainable:
        log.info("froze %d unused backbone pooler tensors", len(trainable))


def build_model(args: argparse.Namespace) -> SentenceTransformer:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        },
    )
    model.max_seq_length = args.max_length

    first_module = model._first_module()
    backbone = getattr(first_module, "auto_model", None)
    if backbone is not None and hasattr(backbone, "config"):
        backbone.config.use_cache = False

    if args.lora:
        try:
            from peft import LoraConfig, TaskType
        except ImportError as error:
            raise RuntimeError("--lora requires the peft package") from error

        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                target_modules="all-linear",
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
        )

    # Run after adapter injection so any pooler adapters are frozen too.
    freeze_unused_pooler(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    log.info(
        "model parameters: trainable=%.2fM total=%.2fM (%.3f%%)",
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / total,
    )
    return model


def main() -> None:
    validate_versions()
    args = parse_args()
    validate_hub_authentication(args)
    torch.manual_seed(args.seed)

    train_dataset, eval_dataset, corpus, corpus_index = prepare_datasets(args)
    if args.prepare_data_only:
        log.info("data preparation completed")
        return

    model = build_model(args)
    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=args.mini_batch_size,
        scale=20.0,
        gather_across_devices=args.gather_across_devices,
        show_progress_bar=False,
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        run_name=args.run_name or Path(args.output).name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        # Do not combine activation checkpointing with a Cached* loss. GradCache
        # already saves memory by replaying mini-batches during backward.
        gradient_checkpointing=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=args.logging_dir or str(Path(args.output) / "tensorboard"),
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,
        hub_private_repo=True if args.hub_private else None,
        hub_always_push=args.hub_always_push,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=args.pin_memory,
        dataloader_persistent_workers=False,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=None,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.push_to_hub:
        # This is blocking: it waits for any asynchronous checkpoint upload,
        # publishes the final model at the repository root, and only then
        # returns. Save the local final/ copy afterwards so it is not uploaded
        # a second time as a nested duplicate.
        trainer.push_to_hub(commit_message="Training complete")
        trainer.save_model(
            str(Path(args.output) / "final"),
            _internal_call=True,
        )
    else:
        trainer.save_model(str(Path(args.output) / "final"))

    del corpus_index, corpus, train_dataset, eval_dataset
    gc.collect()


if __name__ == "__main__":
    main()
