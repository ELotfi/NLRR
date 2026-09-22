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
import inspect
import json
import logging
import math
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

# QUERY_PREFIX = "query: "
# DOC_PREFIX = "passage: "
QUERY_PREFIX = "Query: "
DOC_PREFIX = ""
TASK_DESCRIPTION = "Given a Dutch legal query, retrieve relevant articles that help answer the query."

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



def prep_query(query: str) -> str:
	out = f'Instruct: {TASK_DESCRIPTION}\n' if TASK_DESCRIPTION else ''
	return f'{out}{QUERY_PREFIX}{query}'



def prep_doc(doc: str) -> str:
	return f'{DOC_PREFIX}{doc}'



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
			row = {"anchor": prep_query(q), "positive": prep_doc(corpus[p])}
			for i, nid in enumerate(rng.sample(negs, n_neg)):
				row[f"negative_{i}"] = prep_doc(corpus[nid])

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
		queries[qid] = prep_query(r["query"])
		relevant[qid] = sorted(pos)
		gold.update(pos)
		if len(queries) >= max_queries:
			break

	doc_ids = set(gold)
	pool = [a for a in corpus if a not in gold]
	rng.shuffle(pool)
	doc_ids.update(pool[:max(0, max_docs - len(doc_ids))])
	payload = {"queries": queries, "relevant": relevant,
			   "corpus": {a: prep_doc(corpus[a]) for a in doc_ids}}
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


# ------------------------- VERSION COMPAT ----------------------------------

def compat_training_args(cls, kwargs: dict, main: bool) -> dict:
	"""
	Keep only arguments this transformers version accepts.

	transformers 5.15 REMOVED `logging_dir` and `warmup_ratio` (both now raise
	TypeError). Rather than pin a version, arguments are filtered against the
	installed signature -- and every dropped one is LOGGED, because silently
	losing e.g. push_to_hub would be far worse than a crash.
	"""
	accepted = set(inspect.signature(cls.__init__).parameters)
	# dataclass-based args also expose their fields here
	accepted |= set(getattr(cls, "__dataclass_fields__", {}))
	kept = {k: v for k, v in kwargs.items() if k in accepted}
	dropped = sorted(set(kwargs) - set(kept))
	if dropped and main:
		log.warning("TrainingArguments: dropped unsupported args %s", dropped)
	return kept


def warmup_steps_for(n_rows: int, per_device_bs: int, world: int,
					 epochs: float, ratio: float) -> int:
	"""`warmup_ratio` is gone in transformers >= 5.15; compute the step count
	it used to imply. Approximate is fine -- warmup is not sensitive to +-5%."""
	steps_per_epoch = math.ceil(n_rows / max(1, per_device_bs * world))
	return max(1, round(ratio * steps_per_epoch * epochs))


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


# ---------------------------------------------------------------------------
# Drop-in replacements for train_cl.py. Replace your build_model() with this
# block, and add the four --lora* arguments shown at the bottom to main().
# ---------------------------------------------------------------------------

def freeze_unused_pooler(model) -> list[str]:
	"""
	Freeze the transformer's built-in `pooler` (and any LoRA adapters placed
	on it). e5 MEAN-pools token embeddings, so the XLM-R pooler never enters
	the loss; left trainable, DDP (find_unused_parameters=False) waits for its
	gradient forever -- the "Expected to have finished reduction" error at
	parameter indices 197/198.

	Matches on the path segment ".pooler.", so it also catches LoRA weights
	such as "...pooler.dense.lora_A.default.weight" when target_modules is
	"all-linear".
	"""
	frozen = []
	for name, param in model.named_parameters():
		if ".pooler." in f".{name}." and param.requires_grad:
			param.requires_grad = False
			frozen.append(name)
	if frozen:
		log.info("froze %d unused pooler tensors: %s", len(frozen), frozen)
	return frozen



def build_model(args: argparse.Namespace) -> SentenceTransformer:
	local_rank = int(os.environ.get("LOCAL_RANK", "0"))
	device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		torch.cuda.set_device(local_rank)

	model_kwargs = {"low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
	# FULL FINE-TUNING: keep FP32 master weights. bf16=True in the training
	# args already runs the forward/backward in bf16 via autocast; loading the
	# weights themselves in bf16 means updates of ~lr*1 = 3e-5 are below
	# bf16's rounding step for typical weights (~0.02), so many round to zero
	# and training silently stalls.
	# LoRA: the base is frozen and only small adapters train, so bf16 base
	# weights are fine and save memory.
	if args.lora:
		model_kwargs["torch_dtype"] = torch.bfloat16

	model = SentenceTransformer(args.model, device=device,
								model_kwargs=model_kwargs)
	model.max_seq_length = args.max_len          # was args.max_length (no such arg)

	backbone = getattr(model._first_module(), "auto_model", None)
	if backbone is not None and hasattr(backbone, "config"):
		backbone.config.use_cache = False

	if args.lora:
		try:
			from peft import LoraConfig, TaskType
		except ImportError as error:
			raise RuntimeError("--lora requires the peft package") from error
		model.add_adapter(LoraConfig(
			task_type=TaskType.FEATURE_EXTRACTION,
			inference_mode=False,
			target_modules="all-linear",
			r=args.lora_r,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
		))

	# After adapter injection, so pooler adapters are frozen too.
	freeze_unused_pooler(model)

	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	total = sum(p.numel() for p in model.parameters())
	log.info("model parameters: trainable=%.2fM total=%.2fM (%.3f%%) dtype=%s",
			 trainable / 1e6, total / 1e6, 100.0 * trainable / total,
			 next(model.parameters()).dtype)
	return model


# ---- add to main()'s argparse ---------------------------------------------
#   ap.add_argument("--lora", action="store_true",
#                   help="Not recommended below ~1B params: full FT fits and "
#                        "adapts the representation better.")
#   ap.add_argument("--lora-r", type=int, default=32)
#   ap.add_argument("--lora-alpha", type=int, default=64)
#   ap.add_argument("--lora-dropout", type=float, default=0.05)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", default=os.environ.get("HF_DATASET"))
	ap.add_argument("--corpus-config", default="corpus")
	ap.add_argument("--train-config", default="train")
	ap.add_argument("--model", default="intfloat/multilingual-e5-large-instruct")
	ap.add_argument("--out", default="runs/e5-large-inst")
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
	ap.add_argument("--gather_across_devices", action="store_true",
					help="for CachedMultipleNegativesRankingLoss")
	ap.add_argument("--negatives", type=int, default=8)
	ap.add_argument("--lr", type=float, default=1.5e-5)
	ap.add_argument("--scale", type=float, default=50.0)
	ap.add_argument("--max-len", type=int, default=512)
	ap.add_argument("--val-frac", type=float, default=0.03)
	ap.add_argument("--eval-queries", type=int, default=500)
	ap.add_argument("--eval-docs", type=int, default=20000)
	ap.add_argument("--save-steps", type=int, default=500)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--ignore-weights", action="store_true",
					help="Keep every row regardless of its sample weight.")
	ap.add_argument("--reprepare", action="store_true",
					help="Rebuild prepared data even if it already exists.")
	ap.add_argument("--lora", action="store_true",
					help="Not recommended below ~1B params: full FT fits and "
						"adapts the representation better.")
	ap.add_argument("--lora-r", type=int, default=32)
	ap.add_argument("--lora-alpha", type=int, default=64)
	ap.add_argument("--lora-dropout", type=float, default=0.05)
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
	# model = SentenceTransformer(args.model,
	#                             model_kwargs={"torch_dtype": torch.bfloat16})

	# model.max_seq_length = args.max_len
	model = build_model(args)
	# Stored with the model, so anyone loading it from the Hub gets the right
	# prefixes via encode(..., prompt_name="query"). NOT used during training:
	# the prefixes are already in the data, and applying them twice would
	# prepend "query: query: ".
	model.prompts = {"query": f"Instruct: {TASK_DESCRIPTION}\nQuery: ", "document": ""}

	loss = CachedMultipleNegativesRankingLoss(
		model, mini_batch_size=args.mini_batch, gather_across_devices=args.gather_across_devices, scale=args.scale)

	push = bool(args.hub_model_id)

	# TensorBoard directory: transformers >= 5.15 reads this env var instead of
	# the removed `logging_dir`. Setting both is harmless on older versions.
	tb_dir = str(Path(args.out) / "runs")
	os.environ["TENSORBOARD_LOGGING_DIR"] = tb_dir

	warmup = warmup_steps_for(ds["train"].num_rows, args.batch_size,
							  state.num_processes, args.epochs, 0.1)
	if state.is_main_process:
		log.info("warmup_steps=%d (10%% of ~%d total optimizer steps)", warmup,
				 warmup * 10)

	targs_kw = dict(
		output_dir=args.out,
		num_train_epochs=args.epochs,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		learning_rate=args.lr,
		warmup_steps=warmup,               # replaces warmup_ratio (removed in 5.15)
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

		# ---- logging: TensorBoard --------------------------------------
		report_to=["tensorboard"],
		logging_dir=tb_dir,                # dropped automatically on >= 5.15
		logging_steps=25,
		logging_first_step=True,

		# ---- checkpoints -> Hub -----------------------------------------
		push_to_hub=push,
		hub_model_id=args.hub_model_id if push else None,
		hub_strategy="checkpoint",
		hub_private_repo=not args.hub_public,

		seed=args.seed,
		ddp_find_unused_parameters=False,
	)
	targs = SentenceTransformerTrainingArguments(
		**compat_training_args(SentenceTransformerTrainingArguments, targs_kw,
							   state.is_main_process))
	if push and not getattr(targs, "push_to_hub", False) and state.is_main_process:
		log.warning("push_to_hub was NOT applied -- checkpoints will stay local")

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