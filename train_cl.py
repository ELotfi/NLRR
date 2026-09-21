"""
train_retriever_e5.py
---------------------------------------------------------------------------
Fine-tune intfloat/multilingual-e5-* on the legal retrieval dataset from the
Hugging Face Hub, multi-GPU, pushing checkpoints to the Hub and logging to
TensorBoard.

    source env.sh
    accelerate launch train_retriever_e5.py \
        --dataset $HF_DATASET --hub-model-id $HF_MODEL_REPO

DATA (one Hub repo, two configs)
    corpus : {"id", "text"}
    train  : {"query", "positives", "negatives", "cluster", "weight", ...}

MULTI-GPU
`accelerate launch` runs this whole script once PER GPU. Preparation therefore
happens on rank 0 only, the others wait at a barrier, and everyone then reads
the same memory-mapped files. Without that guard, N processes each built the
corpus dict (N x RAM) and wrote the same files concurrently.

MEMORY
  * the corpus dict exists on rank 0 only, and only during preparation
  * examples are streamed to JSONL and reloaded as memory-mapped Arrow
  * the in-training evaluator uses a capped corpus slice written to disk

E5 CONVENTIONS
  * "query: " on queries, "passage: " on documents -- also stored in the saved
    model as prompts, so whoever loads it from the Hub gets them by default
  * mean pooling, RIGHT padding, max_seq_length 512

CHECKPOINTS -> HUB
hub_strategy="checkpoint" pushes each save AND a resumable `last-checkpoint`
folder. If the rented box dies, resume on a new one with --resume.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
DOC_PREFIX = "passage: "


# ------------------------- PREPARATION (rank 0 only) -----------------------

def cluster_bucket(cluster: str, buckets: int = 100) -> int:
    """Deterministic split by cluster: no set of clusters or list of rows is
    held, and a dispute pattern never straddles train/val."""
    h = hashlib.blake2b(str(cluster).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % buckets


def load_corpus(dataset: str, config: str) -> dict[str, str]:
    ds = load_dataset(dataset, config, split="train")
    corpus = {}
    for rec in ds:                     # Arrow-backed; rows paged in on demand
        t = (rec.get("text") or "").strip()
        if t:
            corpus[rec["id"]] = t
    return corpus


def _neg_ids(negs) -> list[str]:
    """Negatives may arrive as [{"id","tier"}] or as bare ids, depending on how
    the Hub dataset was serialised."""
    out = []
    for n in negs or []:
        if isinstance(n, dict):
            if n.get("id"):
                out.append(n["id"])
        elif n:
            out.append(str(n))
    return out


def write_examples(train_rows, corpus: dict[str, str], n_neg: int,
                   train_out: Path, val_out: Path, val_frac: float,
                   rng: random.Random, use_weights: bool = True) -> dict:
    """
    Stream rows to two JSONL files; nothing accumulates in RAM.

    Sample WEIGHTS are applied by importance sampling: a row with weight w is
    kept with probability w. CachedMNRL has no per-example weight hook, and in
    expectation this is equivalent to scaling that row's gradient.
    """
    stats = {"train": 0, "val": 0, "skipped": 0, "downweighted": 0}
    val_buckets = max(1, int(100 * val_frac))

    with train_out.open("w", encoding="utf-8") as ftr, \
            val_out.open("w", encoding="utf-8") as fva:
        for r in train_rows:
            q = (r.get("query") or "").strip()
            pos = [p for p in (r.get("positives") or []) if p in corpus]
            if not q or not pos:
                stats["skipped"] += 1
                continue
            negs = [n for n in _neg_ids(r.get("negatives"))
                    if n in corpus and n not in set(pos)]
            if len(negs) < n_neg:
                stats["skipped"] += 1
                continue

            w = float(r.get("weight") or 1.0)
            if use_weights and w < 1.0 and rng.random() >= w:
                stats["downweighted"] += 1
                continue

            p = rng.choice(pos)       # one positive per row, see module doc
            row = {"anchor": QUERY_PREFIX + q, "positive": DOC_PREFIX + corpus[p]}
            for i, nid in enumerate(rng.sample(negs, n_neg)):
                row[f"negative_{i}"] = DOC_PREFIX + corpus[nid]

            cluster = r.get("cluster") or r.get("source_case") or q
            if cluster_bucket(cluster) < val_buckets:
                fva.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["val"] += 1
            else:
                ftr.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["train"] += 1
    return stats


def write_eval_slice(train_rows, corpus: dict[str, str], out: Path,
                     max_queries: int, max_docs: int, val_frac: float,
                     rng: random.Random) -> dict:
    """
    Capped IR slice for in-training model selection, WRITTEN TO DISK so every
    rank can load it without a corpus dict. Gold docs always included, topped
    up with random distractors. A relative signal only: the real evaluation is
    the temporal holdout against the full corpus, run separately.
    """
    val_buckets = max(1, int(100 * val_frac))
    queries, relevant, gold = {}, {}, set()
    for r in train_rows:
        cluster = r.get("cluster") or r.get("source_case") or r.get("query")
        if cluster_bucket(cluster) >= val_buckets:
            continue
        pos = [p for p in (r.get("positives") or []) if p in corpus]
        if not pos or not r.get("query"):
            continue
        qid = f"q{len(queries)}"
        queries[qid] = QUERY_PREFIX + r["query"]
        relevant[qid] = sorted(pos)
        gold.update(pos)
        if len(queries) >= max_queries:
            break

    doc_ids = set(gold)
    pool = [a for a in corpus if a not in gold]
    rng.shuffle(pool)
    doc_ids.update(pool[:max(0, max_docs - len(doc_ids))])
    payload = {"queries": queries, "relevant": relevant,
               "corpus": {a: DOC_PREFIX + corpus[a] for a in doc_ids}}
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"queries": len(queries), "docs": len(doc_ids), "gold": len(gold)}


def prepare(args, cache: Path) -> None:
    rng = random.Random(args.seed)
    log.info("loading corpus from %s [%s]", args.dataset, args.corpus_config)
    corpus = load_corpus(args.dataset, args.corpus_config)
    log.info("corpus: %d documents", len(corpus))

    train_ds = load_dataset(args.dataset, args.train_config, split="train")
    log.info("train source: %d rows", train_ds.num_rows)

    stats = write_examples(train_ds, corpus, args.negatives,
                           cache / "train_pairs.jsonl", cache / "val_pairs.jsonl",
                           args.val_frac, rng, use_weights=not args.ignore_weights)
    log.info("prepared %s", stats)

    ev = write_eval_slice(train_ds, corpus, cache / "eval_slice.json",
                          args.eval_queries, args.eval_docs, args.val_frac, rng)
    log.info("eval slice %s", ev)
    (cache / "READY").write_text(json.dumps({**stats, "eval": ev}))
    del corpus, train_ds


# ------------------------- TRAINING (all ranks) ----------------------------

def load_evaluator(path: Path) -> InformationRetrievalEvaluator | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if not d["queries"]:
        return None
    return InformationRetrievalEvaluator(
        queries=d["queries"], corpus=d["corpus"],
        relevant_docs={k: set(v) for k, v in d["relevant"].items()},
        name="legal-val", accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 10], ndcg_at_k=[10], mrr_at_k=[10],
        show_progress_bar=False, batch_size=128, write_csv=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("HF_DATASET"))
    ap.add_argument("--corpus-config", default="corpus")
    ap.add_argument("--train-config", default="train")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", default="runs/e5-base")
    ap.add_argument("--hub-model-id", default=os.environ.get("HF_MODEL_REPO"),
                    help="username/repo to push checkpoints to. Omit to keep local.")
    ap.add_argument("--hub-public", action="store_true",
                    help="Create the Hub repo public (default: private).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint (local, or pulled "
                         "from the Hub's last-checkpoint folder).")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="PER DEVICE. The contrastive-signal knob, not memory.")
    ap.add_argument("--mini-batch", type=int, default=64,
                    help="GradCache chunk: memory only, does not change the loss.")
    ap.add_argument("--negatives", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--eval-queries", type=int, default=500)
    ap.add_argument("--eval-docs", type=int, default=20000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ignore-weights", action="store_true",
                    help="Keep every row regardless of its sample weight.")
    ap.add_argument("--reprepare", action="store_true",
                    help="Rebuild prepared data even if it already exists.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.dataset:
        raise SystemExit("--dataset (or HF_DATASET) is required")

    state = PartialState()
    cache = Path(args.out) / "data"
    ready = cache / "READY"

    # ---- rank 0 prepares; everyone else waits --------------------------
    if state.is_main_process:
        cache.mkdir(parents=True, exist_ok=True)
        if args.reprepare or not ready.exists():
            if ready.exists():
                ready.unlink()
            prepare(args, cache)
        else:
            log.info("reusing prepared data: %s", ready.read_text())
    state.wait_for_everyone()

    ds = load_dataset("json",
                      data_files={"train": str(cache / "train_pairs.jsonl"),
                                  "validation": str(cache / "val_pairs.jsonl")},
                      cache_dir=str(cache / "hf"))
    evaluator = load_evaluator(cache / "eval_slice.json")
    if state.is_main_process:
        log.info("train %d / val %d rows (memory-mapped)",
                 ds["train"].num_rows, ds["validation"].num_rows)

    # ---- model ----------------------------------------------------------
    model = SentenceTransformer(args.model,
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = args.max_len
    # Stored with the model, so anyone loading it from the Hub gets the right
    # prefixes via encode(..., prompt_name="query"). NOT used during training:
    # the prefixes are already in the data, and applying them twice would
    # prepend "query: query: ".
    model.prompts = {"query": QUERY_PREFIX, "passage": DOC_PREFIX}

    loss = CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=args.mini_batch, scale=20.0)

    push = bool(args.hub_model_id)
    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,

        eval_strategy="steps" if evaluator else "no",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=bool(evaluator),
        metric_for_best_model="eval_legal-val_cosine_ndcg@10",
        greater_is_better=True,

        # ---- logging: TensorBoard, under <out>/runs --------------------
        report_to=["tensorboard"],
        logging_dir=str(Path(args.out) / "runs"),
        logging_steps=25,
        logging_first_step=True,

        # ---- checkpoints -> Hub -----------------------------------------
        # "checkpoint" pushes every save plus a resumable `last-checkpoint`
        # folder; the TensorBoard logs travel with it and show up in the
        # Hub's "Training metrics" tab.
        push_to_hub=push,
        hub_model_id=args.hub_model_id if push else None,
        hub_strategy="checkpoint",
        hub_private_repo=not args.hub_public,

        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        loss=loss, evaluator=evaluator)

    resume = None
    if args.resume:
        local = sorted(Path(args.out).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if local:
            resume = str(local[-1])
        elif push:
            # fresh box: pull the resumable checkpoint the Hub kept for us
            from huggingface_hub import snapshot_download
            snap = snapshot_download(args.hub_model_id,
                                     allow_patterns=["last-checkpoint/*"],
                                     local_dir=args.out)
            cand = Path(snap) / "last-checkpoint"
            resume = str(cand) if cand.exists() else None
        if state.is_main_process:
            log.info("resuming from %s", resume or "(nothing found; fresh start)")

    trainer.train(resume_from_checkpoint=resume)

    # ---- final model ----------------------------------------------------
    final = Path(args.out) / "final"
    if state.is_main_process:
        model.save_pretrained(str(final))
        log.info("saved final model to %s", final)
    if push:
        # Pushes the best model (load_best_model_at_end) with a model card,
        # the prompts above, and the TensorBoard logs.
        trainer.push_to_hub(commit_message="final model")
        if state.is_main_process:
            log.info("pushed to https://huggingface.co/%s", args.hub_model_id)


if __name__ == "__main__":
    main()