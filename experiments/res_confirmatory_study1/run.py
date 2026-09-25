"""Bounded, checkpointed Qwen inference for the frozen RES study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import protocol as p

RESULTS = p.ROOT / "results"


def now():
    return datetime.now(timezone.utc).isoformat()


def shard_path(index):
    return RESULTS / "shards" / f"shard{index:02d}.json"


def validate(payload, index, freeze_hash):
    expected = p.rendered()[index * p.SHARD_SIZE:(index + 1) * p.SHARD_SIZE]
    for key, value in dict(protocol=p.VERSION, freeze_hash=freeze_hash, model=p.MODEL,
                           revision=p.REVISION, shard=index, backend="transformers-cpu",
                           dtype="torch.bfloat16").items():
        if payload.get(key) != value:
            raise ValueError(f"Shard {index}: mismatched {key}")
    records = payload["records"]
    if len(records) not in (0, 4, 8):
        raise ValueError("Only full four-prompt checkpoints are valid")
    if payload["status"] != ("COMPLETE" if len(records) == 8 else "PARTIAL"):
        raise ValueError("Incorrect checkpoint status")
    for r, row in zip(records, expected):
        if r["observation_id"] != row["observation_id"] or r["messages_sha256"] != row["messages_sha256"]:
            raise ValueError("Fixture/prompt order mismatch")
        if r["parsed"] != p.parse_choice(r["text"]) or r["passed"] != (r["parsed"] == row["expected"]):
            raise ValueError("Frozen parser or scoring mismatch")
        if r["generated_classification"] != r["parsed"]:
            raise ValueError("Generated classification mismatch")
        if r["logit_classification"] != ("A" if r["raw_ab_margin"] >= 0 else "B"):
            raise ValueError("Frozen A/B logit convention mismatch")
        if not 0 < r["prompt_token_count"] <= p.MAX_INPUT or not 0 < len(r["generated_ids"]) <= p.MAX_NEW:
            raise ValueError("Input or output token cap exceeded")
        if not 0 <= r["response_token_count"] <= p.MAX_NEW or not math.isfinite(r["raw_ab_margin"]):
            raise ValueError("Invalid output length or margin")
        if not all(math.isfinite(r[k]) for k in ("raw_a_logit", "raw_b_logit", "processed_a_logit", "processed_b_logit")):
            raise ValueError("Invalid raw/processed logits")
        if r["first_generated_id"] != r["generated_ids"][0] or r["prompt_char_count"] != row["prompt_char_count"]:
            raise ValueError("Raw generation or prompt count mismatch")
    return records


def load(index, freeze_hash):
    path = shard_path(index)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    validate(payload, index, freeze_hash)
    return payload


def tokenizer_audit(freeze_hash):
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL, revision=p.REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = []
    for row in p.rendered():
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if not 0 < n_tokens <= p.MAX_INPUT:
            raise ValueError(f"{row['observation_id']}: {n_tokens} input tokens exceeds {p.MAX_INPUT}")
        rows.append(dict(observation_id=row["observation_id"], messages_sha256=row["messages_sha256"],
                         chat_template_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                         prompt_token_count=n_tokens, chat_template_char_count=len(prompt),
                         prompt_char_count=row["prompt_char_count"]))
    codes = [tokenizer.encode(x, add_special_tokens=False) for x in "AB"]
    if any(len(ids) != 1 for ids in codes) or codes[0] == codes[1]:
        raise ValueError("A/B are not distinct single tokenizer tokens")
    result = dict(protocol=p.VERSION, freeze_hash=freeze_hash, model=p.MODEL, revision=p.REVISION,
                  transformers=transformers.__version__, a_token_id=codes[0][0], b_token_id=codes[1][0],
                  rows=rows)
    p.atomic_json(RESULTS / "token_counts.json", result)
    return result


def run_shard(index, freeze_hash):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    existing = load(index, freeze_hash)
    if existing and existing["status"] == "COMPLETE":
        print(f"Shard {index} already complete; no inference")
        return
    torch.manual_seed(p.SEED)
    np.random.seed(p.SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL, revision=p.REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = p.rendered()[index * p.SHARD_SIZE:(index + 1) * p.SHARD_SIZE]
    prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
               for r in rows]
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in prompts]
    if max(lengths) > p.MAX_INPUT:
        raise ValueError("Input token cap exceeded before model inference")
    codes = [tokenizer.encode(s, add_special_tokens=False) for s in "AB"]
    if any(len(ids) != 1 for ids in codes) or codes[0] == codes[1]:
        raise ValueError("A/B tokenizer gate failed")
    model = AutoModelForCausalLM.from_pretrained(p.MODEL, revision=p.REVISION, torch_dtype=torch.bfloat16)
    model.eval()
    started = time.monotonic()
    payload = existing or dict(protocol=p.VERSION, freeze_hash=freeze_hash, model=p.MODEL,
                               revision=p.REVISION, shard=index, backend="transformers-cpu",
                               dtype="torch.bfloat16", status="PARTIAL", records=[], attempts=[])
    payload["attempts"].append(dict(run_id=os.environ.get("GITHUB_RUN_ID", "local"), started=now(),
                                    python=platform.python_version(), torch=torch.__version__,
                                    transformers=transformers.__version__, cpu=platform.processor(),
                                    threads=torch.get_num_threads(), generation_config=model.generation_config.to_dict(),
                                    sampling=dict(do_sample=False, max_new_tokens=p.MAX_NEW, seed=p.SEED)))
    p.atomic_json(shard_path(index), payload)
    for start in range(len(payload["records"]), len(rows), 4):
        if time.monotonic() - started >= 4800:
            raise TimeoutError("Soft time limit; preserved valid partial checkpoint")
        encoded = tokenizer(prompts[start:start + 4], return_tensors="pt", padding=True,
                            truncation=False, add_special_tokens=False)
        batch_started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, max_new_tokens=p.MAX_NEW,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                    return_dict_in_generate=True, output_logits=True,
                                    output_scores=True, num_logits_to_keep=1)
        elapsed = time.monotonic() - batch_started
        tokens = output.sequences[:, encoded["input_ids"].shape[1]:]
        raw, processed = output.logits[0].float().cpu(), output.scores[0].float().cpu()
        for offset, text in enumerate(tokenizer.batch_decode(tokens, skip_special_tokens=True)):
            i = start + offset
            ids = tokens[offset].cpu().tolist()
            a, b = codes[0][0], codes[1][0]
            margin = float(raw[offset, a] - raw[offset, b])
            parsed = p.parse_choice(text)
            # Response-token count excludes generated EOS and padding; IDs remain intact separately.
            effective = next((j for j, token in enumerate(ids) if token == tokenizer.eos_token_id), len(ids))
            payload["records"].append(dict(
                observation_id=rows[i]["observation_id"], fixture_id=rows[i]["fixture_id"],
                condition=rows[i]["condition"], messages_sha256=rows[i]["messages_sha256"],
                prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(),
                prompt_char_count=rows[i]["prompt_char_count"], prompt_token_count=lengths[i],
                response_token_count=effective, batch_padded_input_tokens=encoded["input_ids"].shape[1],
                text=text, generated_ids=ids, parsed=parsed, generated_classification=parsed,
                passed=(parsed == rows[i]["expected"]),
                raw_a_logit=float(raw[offset, a]), raw_b_logit=float(raw[offset, b]),
                processed_a_logit=float(processed[offset, a]), processed_b_logit=float(processed[offset, b]),
                raw_ab_margin=margin, decision_margin=(margin if rows[i]["expected"] == "A" else -margin),
                logit_classification="A" if margin >= 0 else "B",
                logit_correct=("A" if margin >= 0 else "B") == rows[i]["expected"],
                a_token_id=a, b_token_id=b, first_generated_id=ids[0],
                first_raw_argmax_id=int(raw[offset].argmax()), batch_seconds=elapsed,
                model=p.MODEL, revision=p.REVISION, seed=p.SEED,
                sampling=dict(do_sample=False, max_new_tokens=p.MAX_NEW),
                run_id=os.environ.get("GITHUB_RUN_ID", "local"),
                commit=os.environ.get("GITHUB_SHA", "local"), completed_at=now(),
            ))
        payload["status"] = "COMPLETE" if len(payload["records"]) == p.SHARD_SIZE else "PARTIAL"
        payload["updated"] = now()
        validate(payload, index, freeze_hash)
        p.atomic_json(shard_path(index), payload)
        print(f"Shard {index}: {len(payload['records'])}/8 persisted; batch {elapsed:.1f}s", flush=True)


def collect(artifacts, freeze_hash):
    source = Path(artifacts)
    status = {}
    for index in range(4 * p.N // p.SHARD_SIZE):
        candidates = list(source.rglob(f"shard{index:02d}.json"))
        if len(candidates) > 1:
            raise ValueError(f"Duplicate shard artifacts {index}")
        old = load(index, freeze_hash)
        if candidates:
            new = json.loads(candidates[0].read_text())
            records = validate(new, index, freeze_hash)
            if records[:len((old or {}).get("records", []))] != (old or {}).get("records", []):
                raise ValueError("Refusing to overwrite existing raw observations")
            p.atomic_json(shard_path(index), new)
        status[str(index)] = (load(index, freeze_hash) or {}).get("status", "MISSING")
    token_audits = list(source.rglob("token_counts.json"))
    if len(token_audits) != 1:
        raise ValueError("Exactly one tokenizer audit is required")
    token_data = json.loads(token_audits[0].read_text())
    if token_data["freeze_hash"] != freeze_hash or len(token_data["rows"]) != 4 * p.N:
        raise ValueError("Tokenizer audit mismatch")
    p.atomic_json(RESULTS / "token_counts.json", token_data)
    complete = all(v == "COMPLETE" for v in status.values())
    p.atomic_json(RESULTS / "run_status.json", dict(status="COMPLETE" if complete else "INCOMPLETE",
                                                    shards=status, freeze_hash=freeze_hash,
                                                    run_id=os.environ.get("GITHUB_RUN_ID", "local")))
    print(f"Collected {sum(v == 'COMPLETE' for v in status.values())}/32 complete shards")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "preflight", "token-audit", "run", "collect"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--artifacts", default="incoming")
    args = parser.parse_args()
    if args.command == "freeze":
        print(p.freeze())
        return
    freeze_hash = p.check_freeze()
    if args.command == "preflight":
        print(json.dumps(dict(status="VALIDATED_NO_MODEL_INFERENCE", freeze_hash=freeze_hash, audit=p.audit())))
    elif args.command == "token-audit":
        print(json.dumps(dict(rows=len(tokenizer_audit(freeze_hash)["rows"]), freeze_hash=freeze_hash)))
    elif args.command == "run":
        if args.shard is None or not 0 <= args.shard < 32:
            raise ValueError("Invalid shard index")
        run_shard(args.shard, freeze_hash)
    elif args.command == "collect":
        collect(args.artifacts, freeze_hash)


if __name__ == "__main__":
    main()
