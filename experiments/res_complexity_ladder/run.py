"""Freeze, preflight, plan, checkpoint, and collect bounded RES CPU runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import protocol as p


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate(payload, k, shard, freeze):
    expected = p.shard_rows(k, shard)
    if any(payload.get(key) != value for key, value in dict(
        protocol=p.VERSION, freeze_hash=freeze, model=p.MODEL, revision=p.REVISION,
        dtype="torch.bfloat16", backend="transformers-cpu", factor_count=k, shard=shard,
    ).items()):
        raise ValueError("Checkpoint provenance mismatch")
    records = payload["records"]
    if len(records) not in (0, 4, 8):
        raise ValueError("Only whole, fixed four-row batches are resumable")
    for record, fixture in zip(records, expected):
        if record["fixture"] != fixture:
            raise ValueError("Checkpoint fixture/order mismatch")
        if record["parsed"] != p.parse_choice(record["text"]):
            raise ValueError("Checkpoint parser mismatch")
        if record["generated_correct"] != (record["parsed"] == fixture["ideal"]):
            raise ValueError("Checkpoint correctness mismatch")
        strict = record["text"].strip().upper()
        if record["strict_parse"] != (strict if strict in ("A", "B") else None):
            raise ValueError("Checkpoint strict parser mismatch")
        if not 0 < record["input_tokens"] <= p.MAX_INPUT or not 0 < len(record["generated_ids"]) <= p.MAX_NEW:
            raise ValueError("Token budget mismatch")
        if not math.isfinite(record["raw_ab_margin"]) or not math.isfinite(record["processed_ab_margin"]):
            raise ValueError("Non-finite A/B margin")
        choice = "A" if record["raw_ab_margin"] >= 0 else "B"
        if record["logit_choice"] != choice or record["logit_correct"] != (choice == fixture["ideal"]):
            raise ValueError("Checkpoint logit mismatch")
        for key in ("run_id", "commit", "batch_seconds", "prompt_sha256", "first_generated_id", "first_raw_argmax_id"):
            if key not in record:
                raise ValueError(f"Missing trace field: {key}")
        if record["first_generated_id"] != record["generated_ids"][0]:
            raise ValueError("Generated token trace mismatch")
        if not record["input_tokens"] <= record["batch_padded_input_tokens"] <= p.MAX_INPUT:
            raise ValueError("Padded token budget mismatch")
    if payload["status"] != ("COMPLETE" if len(records) == 8 else "PARTIAL"):
        raise ValueError("Checkpoint status mismatch")
    return records


def load_checkpoint(k, shard, freeze, base=None):
    path = p.shard_path(k, shard, base)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    validate(payload, k, shard, freeze)
    return payload


def stage_complete(k, freeze):
    return all((load_checkpoint(k, s, freeze) or {}).get("status") == "COMPLETE"
               for s in range(p.COUNTS[k] * 4 // p.SHARD_ROWS))


def plan(control, freeze):
    if set(control) != {"operation", "factor_count", "slice", "attempt"}:
        raise ValueError("Unexpected control fields")
    if control["operation"] not in ("preflight", "run", "report"):
        raise ValueError("Unknown operation")
    k, part = control["factor_count"], control["slice"]
    if type(k) is not int or type(part) is not int or k not in p.COUNTS:
        raise ValueError("Invalid factor count or slice")
    if not 0 <= part < p.COUNTS[k] // 8:
        raise ValueError("Invalid slice")
    if type(control["attempt"]) is not int or control["attempt"] < 1:
        raise ValueError("Invalid attempt nonce")
    shards = list(range(part * 4, (part + 1) * 4))
    if control["operation"] == "run":
        for earlier in range(2, k):
            if not stage_complete(earlier, freeze):
                raise ValueError(f"k{earlier} must be complete before k{k}; accuracy is NOT an advancement gate")
        for s in range(part * 4):
            if (load_checkpoint(k, s, freeze) or {}).get("status") != "COMPLETE":
                raise ValueError("Complete the previous slice first")
        shards = [s for s in shards if (load_checkpoint(k, s, freeze) or {}).get("status") != "COMPLETE"]
        if not shards:
            # Avoid empty matrices and even dependency installation for completed work.
            return dict(control, operation="report", shards=[])
    return dict(control, shards=shards)


def run_shard(k, shard, freeze, soft_seconds=4800):
    started = time.monotonic()
    existing = load_checkpoint(k, shard, freeze)
    if existing and existing["status"] == "COMPLETE":
        print("Validated complete shard; zero inference.", flush=True)
        return True
    # Heavy dependencies imported only by actual model execution.
    import hashlib
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(p.SEED)
    np.random.seed(p.SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL, revision=p.REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = p.shard_rows(k, shard)
    prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in prompts]
    if max(lengths) > p.MAX_INPUT:
        raise ValueError(f"Input cap exceeded: {max(lengths)} > {p.MAX_INPUT}; no truncation or inference")
    codes = [tokenizer.encode(s, add_special_tokens=False) for s in "AB"]
    if any(len(ids) != 1 for ids in codes) or codes[0] == codes[1]:
        raise ValueError("A/B are not distinct single tokens")
    model = AutoModelForCausalLM.from_pretrained(p.MODEL, revision=p.REVISION, torch_dtype=torch.bfloat16)
    model.eval()
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       transformers=transformers.__version__, cpu=platform.processor(),
                       cpu_count=os.cpu_count(), threads=torch.get_num_threads(),
                       generation_config=model.generation_config.to_dict(), platform=platform.platform(),
                       explicit_generation_overrides=dict(do_sample=False, max_new_tokens=p.MAX_NEW,
                                                          use_cache=True, batch_size=p.BATCH_SIZE,
                                                          padding_side="left", add_special_tokens=False))
    payload = existing or dict(protocol=p.VERSION, freeze_hash=freeze, model=p.MODEL,
                              revision=p.REVISION, dtype="torch.bfloat16", backend="transformers-cpu",
                              factor_count=k, shard=shard, records=[], status="PARTIAL", attempts=[])
    payload["attempts"].append(dict(started=utc(), run_id=os.environ.get("GITHUB_RUN_ID", "local"),
                                    run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", "1"), environment=environment))
    p.atomic_json(p.shard_path(k, shard), payload)
    for start in range(len(payload["records"]), len(rows), p.BATCH_SIZE):
        if time.monotonic() - started >= soft_seconds:
            print("Soft budget reached; checkpoint retained for resumption.", flush=True)
            return False
        encoded = tokenizer(prompts[start:start + p.BATCH_SIZE], return_tensors="pt", padding=True,
                            truncation=False, add_special_tokens=False)
        batch_started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, max_new_tokens=p.MAX_NEW,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                    return_dict_in_generate=True, output_logits=True,
                                    output_scores=True, num_logits_to_keep=1)
        elapsed = time.monotonic() - batch_started
        tokens = output.sequences[:, encoded["input_ids"].shape[1]:]
        raw = output.logits[0].float().cpu()
        processed = output.scores[0].float().cpu()
        new_records = []
        for offset, text in enumerate(tokenizer.batch_decode(tokens, skip_special_tokens=True)):
            i = start + offset
            margin = (raw[offset, codes[0][0]] - raw[offset, codes[1][0]]).item()
            parsed = p.parse_choice(text)
            logit_choice = "A" if margin >= 0 else "B"  # Frozen tie convention; ties reported.
            ids = tokens[offset].cpu().tolist()
            new_records.append(dict(
                fixture=rows[i], text=text, parsed=parsed, generated_correct=parsed == rows[i]["ideal"],
                strict_parse=text.strip().upper() if text.strip().upper() in ("A", "B") else None,
                raw_ab_margin=margin,
                processed_ab_margin=(processed[offset, codes[0][0]] - processed[offset, codes[1][0]]).item(),
                logit_choice=logit_choice, logit_correct=logit_choice == rows[i]["ideal"],
                first_raw_argmax_id=raw[offset].argmax().item(), first_generated_id=ids[0],
                generated_ids=ids, input_tokens=lengths[i], batch_padded_input_tokens=encoded["input_ids"].shape[1],
                prompt_sha256=hashlib.sha256(prompts[i].encode()).hexdigest(),
                batch_seconds=elapsed, run_id=os.environ.get("GITHUB_RUN_ID", "local"),
                commit=os.environ.get("GITHUB_SHA", "local"), a_token_id=codes[0][0], b_token_id=codes[1][0],
            ))
        payload["records"].extend(new_records)
        payload["status"] = "COMPLETE" if len(payload["records"]) == len(rows) else "PARTIAL"
        payload["updated"] = utc()
        payload["attempts"][-1].update(last_checkpoint=payload["updated"], elapsed_seconds=time.monotonic() - started)
        validate(payload, k, shard, freeze)
        p.atomic_json(p.shard_path(k, shard), payload)
        print(f"k{k} shard {shard}: {len(payload['records'])}/8 saved; batch {elapsed:.1f}s", flush=True)
    return True


def collect(k, part, artifacts, freeze):
    selected = range(part * 4, (part + 1) * 4)
    for shard in selected:
        candidates = list(Path(artifacts).rglob(p.shard_path(k, shard).name))
        if len(candidates) > 1:
            raise ValueError("Duplicate artifacts for one shard")
        old = load_checkpoint(k, shard, freeze)
        if candidates:
            incoming = json.loads(candidates[0].read_text())
            records = validate(incoming, k, shard, freeze)
            old_records = (old or {}).get("records", [])
            if records[:len(old_records)] != old_records:
                raise ValueError("Refusing to overwrite a completed observation")
            p.atomic_json(p.shard_path(k, shard), incoming)
    complete = all((load_checkpoint(k, s, freeze) or {}).get("status") == "COMPLETE" for s in selected)
    p.atomic_json(p.ROOT / "results" / f"k{k}-slice{part}.json", dict(
        protocol=p.VERSION, freeze_hash=freeze, factor_count=k, slice=part,
        status="COMPLETE" if complete else "INCOMPLETE", run_id=os.environ.get("GITHUB_RUN_ID", "local"),
        shards={str(s): (load_checkpoint(k, s, freeze) or {}).get("status", "MISSING") for s in selected},
    ))
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "preflight", "plan", "run", "collect"))
    parser.add_argument("--k", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--slice", type=int)
    parser.add_argument("--artifacts", default="incoming")
    args = parser.parse_args()
    if os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1":
        raise ValueError("Resume with a fresh control.json attempt commit, not GitHub Re-run: the old SHA can omit checkpoints.")
    if args.command == "freeze":
        if (p.ROOT / "manifest.json").exists() or list((p.ROOT / "results").glob("shards/*.json")):
            raise ValueError("Refusing to refreeze existing protocol/results")
        p.audit()
        p.atomic_json(p.ROOT / "manifest.json", p.make_manifest())
        return
    freeze = p.check_freeze()
    if args.command == "preflight":
        print(json.dumps(dict(status="PREFLIGHT_ONLY_NO_INFERENCE", freeze_hash=freeze, audit=p.audit()), indent=2))
    elif args.command == "plan":
        selected = plan(json.loads((p.ROOT / "control.json").read_text()), freeze)
        print(json.dumps(selected))
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                for key in ("operation", "factor_count", "slice"):
                    output.write(f"{key}={selected[key]}\n")
                output.write("matrix=" + json.dumps(dict(shard=selected["shards"])) + "\n")
    elif args.command == "run":
        if not run_shard(args.k, args.shard, freeze):
            sys.exit(2)
    elif args.command == "collect":
        complete = collect(args.k, args.slice, args.artifacts, freeze)
        print("Slice COMPLETE" if complete else "Slice INCOMPLETE; valid checkpoints retained")
        # Reporting/persistence must still execute; the workflow marks incomplete afterward.


if __name__ == "__main__":
    main()
