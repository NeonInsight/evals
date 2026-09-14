#!/usr/bin/env python3
"""Run one logical 16-row shard of the Qwen2.5-3B interface gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_interface_gate as gate

ROOT = Path("experiments/res_step0_interface_gate")
OUT_DIR = Path(os.environ.get("RES_INTERFACE_OUTPUT_DIR", ROOT / "results"))
AMENDMENT = ROOT / "preregistration-qwen3b-execution-amendment.md"
SEQUENCE_ID = os.environ.get("RES_INTERFACE_SEQUENCE_ID", "qwen3b-gate-v2")
SHARD_ID = os.environ["RES_INTERFACE_SHARD_ID"]
SHARDS = {
    "five-key-rotation": ("five_factor", "key-rotation"),
    "five-schedule-change": ("five_factor", "schedule-change"),
    "minimal-release-approval": ("minimal_owner_indexed", "release-approval"),
    "minimal-account-change": ("minimal_owner_indexed", "account-change"),
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if SHARD_ID not in SHARDS or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", SEQUENCE_ID):
        raise ValueError("Invalid fixed shard or sequence identifier")
    started = datetime.now(timezone.utc)
    random.seed(gate.SEED)
    np.random.seed(gate.SEED)
    torch.manual_seed(gate.SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    five, minimal = gate.build_five_factor_rows(), gate.build_minimal_rows()
    gate.validate_rows(five, "five_factor")
    gate.validate_rows(minimal, "minimal_owner_indexed")
    datasets = {"five_factor": five, "minimal_owner_indexed": minimal}
    dataset_hash = digest(gate.json_bytes(datasets))
    condition, family = SHARDS[SHARD_ID]
    selected = [(i, row) for i, row in enumerate(datasets[condition]) if row["family_id"] == family]
    ideals = [row["ideal"] for _, row in selected]
    if len(selected) != 16 or ideals.count("A") != 8 or ideals.count("B") != 8:
        raise AssertionError("Shard must contain sixteen label-balanced rows")

    print(f"Running {SHARD_ID}: 16 frozen rows.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(gate.MODEL_ID, revision=gate.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    action_ids = gate.choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        gate.MODEL_ID, revision=gate.MODEL_REVISION, torch_dtype=gate.MODEL_DTYPE
    )
    model.eval()

    fixtures = []
    prompts = [gate.prompt_for(row, tokenizer) for _, row in selected]
    batches = list(gate.batched(prompts, gate.BATCH_SIZE))
    for batch_no, (start, prompt_batch) in enumerate(batches, 1):
        encoded = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=False)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=4,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
                num_logits_to_keep=1,
            )
        raw_logits = output.logits
        if not raw_logits:
            raise RuntimeError("Generation did not return raw first-step logits")
        margins = raw_logits[0][:, action_ids[0]] - raw_logits[0][:, action_ids[1]]
        generated_tokens = output.sequences[:, encoded["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for offset, (text, margin) in enumerate(zip(texts, margins.detach().float().cpu().tolist())):
            index, row = selected[start + offset]
            prompt = prompt_batch[offset]
            fixtures.append({
                "row_id": f"{condition}:{index:02d}",
                "condition": condition,
                "family_id": family,
                "ideal": row["ideal"],
                "rendered_prompt": prompt,
                "prompt_sha256": digest(prompt.encode("utf-8")),
                "logit_choice": "A" if margin >= 0 else "B",
                "logit_margin_a_minus_b": float(margin),
                "generated_raw": text,
                "generated_choice": gate.parse_choice(text),
            })
        print(f"Completed batch {batch_no}/{len(batches)}.", flush=True)

    finished = datetime.now(timezone.utc)
    revision = getattr(model.config, "_commit_hash", None) or gate.MODEL_REVISION
    runner_path = ROOT / "run_interface_gate_shard.py"
    result = {
        "schema_version": "res-step0-interface-gate-shard-v1",
        "status": "COMPLETED_SHARD",
        "sequence_id": SEQUENCE_ID,
        "shard_id": SHARD_ID,
        "condition": condition,
        "family_id": family,
        "run_id": gate.RUN_ID,
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "model": {
            "id": gate.MODEL_ID,
            "requested_revision": gate.MODEL_REVISION,
            "resolved_revision": revision,
            "requested_dtype": gate.MODEL_DTYPE_NAME,
            "dtype": str(model.dtype),
            "action_token_ids": {"A": action_ids[0], "B": action_ids[1]},
        },
        "execution": {
            "seed": gate.SEED,
            "batch_size": gate.BATCH_SIZE,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "combined_generation_and_logit_pass": True,
            "num_logits_to_keep": 1,
        },
        "artifacts": {
            "full_dataset_sha256": dataset_hash,
            "original_preregistration_sha256": digest(gate.PREREG_PATH.read_bytes()),
            "execution_amendment_sha256": digest(AMENDMENT.read_bytes()),
            "base_runner_sha256": digest(gate.RUNNER_PATH.read_bytes()),
            "shard_runner_sha256": digest(runner_path.read_bytes()),
            "fixtures": len(fixtures),
        },
        "fixtures": fixtures,
    }
    target = OUT_DIR / "shards" / SEQUENCE_ID / f"{SHARD_ID}.json"
    write_json(target, result)
    print(json.dumps({"status": result["status"], "result": str(target)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
