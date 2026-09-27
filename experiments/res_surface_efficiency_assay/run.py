#!/usr/bin/env python3
"""Hash-bound, resumable execution for the frozen surface-efficiency assay."""
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
from typing import Any

import study

RESULTS = study.ROOT / "results"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_hashes() -> str:
    manifest_path = study.ROOT / "frozen_manifest.json"
    hashes_path = study.ROOT / "hashes.sha256"
    if not manifest_path.exists() or not hashes_path.exists():
        raise ValueError("Phase 0 freeze artifacts are absent")
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = study.REPO / relative
        if study.file_hash(path) != expected:
            raise ValueError(f"Frozen file hash mismatch: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != study.VERSION:
        raise ValueError("Frozen manifest protocol changed")
    if manifest.get("model_identifier") != study.MODEL_ID or manifest.get("model_revision") != study.MODEL_REVISION:
        raise ValueError("Frozen model binding changed")
    return study.file_hash(manifest_path)


def manifest() -> dict[str, Any]:
    return json.loads((study.ROOT / "frozen_manifest.json").read_text(encoding="utf-8"))


def renderings() -> list[dict[str, Any]]:
    return study.read_jsonl(study.ROOT / "rendered_prompts.jsonl")


def fixtures() -> list[dict[str, Any]]:
    return study.read_jsonl(study.ROOT / "fixtures.jsonl")


def shard_count() -> int:
    return len(fixtures()) // study.SHARD_FIXTURES


def shard_path(index: int) -> Path:
    return RESULTS / "shards" / f"shard{index:02d}.json"


def rows_for_shard(index: int) -> list[dict[str, Any]]:
    selected_ids = {
        fixture["fixture_id"]
        for fixture in fixtures()
        if fixture["shard_index"] == index
    }
    rows = [row for row in renderings() if row["fixture_id"] in selected_ids]
    rows.sort(key=lambda row: row["evaluation_order"])
    expected_rows = study.SHARD_FIXTURES * len(study.CONDITIONS)
    if len(selected_ids) != study.SHARD_FIXTURES or len(rows) != expected_rows:
        raise ValueError("Shard does not contain complete frozen fixtures")
    return rows


def validate_observation(record: dict[str, Any], specification: dict[str, Any], frozen: dict[str, Any]) -> None:
    bindings = {
        "observation_id": specification["observation_id"],
        "fixture_id": specification["fixture_id"],
        "condition": specification["condition"],
        "rendering_hash": specification["rendering_hash"],
        "messages_sha256": specification["messages_sha256"],
        "model": frozen["model_identifier"],
        "revision": frozen["model_revision"],
        "inference_config_hash": frozen["inference_config_hash"],
        "scorer_config_hash": frozen["scorer_hash"],
    }
    for key, expected in bindings.items():
        if record.get(key) != expected:
            raise ValueError(f"Stored observation has mismatched {key}; refusing reuse")
    status = record.get("technical_status")
    if status not in {"VALID", "TECHNICAL_INVALID"}:
        raise ValueError("Unknown technical status")
    if status == "VALID":
        if record["parsed_choice"] != study.parse_choice(record["text"]):
            raise ValueError("Stored parser result differs from frozen parser")
        if record["passed"] != (record["parsed_choice"] == specification["expected"]):
            raise ValueError("Stored deterministic score differs")
        if not math.isfinite(record["decision_margin"]):
            raise ValueError("Stored decision margin is non-finite")
    if record["attempt_count"] > 1 + study.TECHNICAL_RETRIES:
        raise ValueError("Stored observation exceeds frozen retry limit")


def validate_shard(payload: dict[str, Any], index: int, freeze_hash: str) -> None:
    frozen = manifest()
    expected_rows = rows_for_shard(index)
    expected_by_fixture: dict[str, list[dict[str, Any]]] = {}
    for row in expected_rows:
        expected_by_fixture.setdefault(row["fixture_id"], []).append(row)
    expected_fixture_ids = sorted(expected_by_fixture, key=lambda fid: min(r["evaluation_order"] for r in expected_by_fixture[fid]))
    bindings = {
        "protocol": study.VERSION,
        "freeze_manifest_hash": freeze_hash,
        "model": study.MODEL_ID,
        "revision": study.MODEL_REVISION,
        "inference_config_hash": frozen["inference_config_hash"],
        "scorer_config_hash": frozen["scorer_hash"],
        "shard_index": index,
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"Shard {index} mismatched {key}; refusing reuse")
    completed = payload.get("completed_fixture_ids", [])
    if completed != expected_fixture_ids[: len(completed)]:
        raise ValueError("Shard fixture checkpoint is not a frozen-order prefix")
    records = payload.get("records", [])
    if len(records) != len(study.CONDITIONS) * len(completed):
        raise ValueError("Checkpoint does not contain every frozen condition per completed fixture")
    record_index = {record["observation_id"]: record for record in records}
    if len(record_index) != len(records):
        raise ValueError("Duplicate observation in shard")
    for fixture_id in completed:
        for specification in expected_by_fixture[fixture_id]:
            if specification["observation_id"] not in record_index:
                raise ValueError("Completed fixture lacks a frozen observation")
            validate_observation(record_index[specification["observation_id"]], specification, frozen)
    expected_status = "COMPLETE" if len(completed) == study.SHARD_FIXTURES else "PARTIAL"
    if payload.get("status") != expected_status:
        raise ValueError("Shard status does not match its fixture checkpoint")


def load_shard(index: int, freeze_hash: str) -> dict[str, Any] | None:
    path = shard_path(index)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_shard(payload, index, freeze_hash)
    return payload


def token_audit(freeze_hash: str) -> dict[str, Any]:
    import transformers
    from transformers import AutoTokenizer

    frozen = manifest()
    tokenizer = AutoTokenizer.from_pretrained(study.MODEL_ID, revision=study.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    audited = []
    for row in renderings():
        prompt = study.chat_prompt(tokenizer, row["messages"])
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if len(tokens) != row["input_token_count"] or len(prompt) != row["character_count"]:
            raise ValueError("Frozen tokenizer length audit changed")
        if prompt_hash != row["chat_prompt_sha256"]:
            raise ValueError("Frozen chat-template prompt hash changed")
        if not 0 < len(tokens) <= study.MAX_INPUT_TOKENS:
            raise ValueError("Frozen input-token limit exceeded")
        audited.append(
            {
                "observation_id": row["observation_id"],
                "rendering_hash": row["rendering_hash"],
                "chat_prompt_sha256": prompt_hash,
                "input_token_count": len(tokens),
                "character_count": len(prompt),
            }
        )
    code_ids = {choice: tokenizer.encode(choice, add_special_tokens=False) for choice in "AB"}
    if any(len(ids) != 1 for ids in code_ids.values()) or code_ids["A"] == code_ids["B"]:
        raise ValueError("A/B are not distinct single admissible decision tokens")
    payload = {
        "protocol": study.VERSION,
        "freeze_manifest_hash": freeze_hash,
        "model": study.MODEL_ID,
        "revision": study.MODEL_REVISION,
        "transformers": transformers.__version__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "decision_token_ids": {key: value[0] for key, value in code_ids.items()},
        "inference_config_hash": frozen["inference_config_hash"],
        "scorer_config_hash": frozen["scorer_hash"],
        "rows": audited,
    }
    study.atomic_json(RESULTS / "token_audit.json", payload)
    return payload


def _decode_attempt(model: Any, tokenizer: Any, rows: list[dict[str, Any]], prompts: list[str]) -> list[dict[str, Any]]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=study.MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            num_logits_to_keep=1,
        )
    if not output.logits:
        raise RuntimeError("Generation did not return first-step raw logits")
    tokens = output.sequences[:, encoded["input_ids"].shape[1] :]
    raw = output.logits[0].detach().float().cpu()
    processed = output.scores[0].detach().float().cpu()
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
    results = []
    for index, text in enumerate(tokenizer.batch_decode(tokens, skip_special_tokens=True)):
        ids = tokens[index].detach().cpu().tolist()
        effective = next((position for position, token in enumerate(ids) if token == tokenizer.eos_token_id), len(ids))
        expected_id = a_id if rows[index]["expected"] == "A" else b_id
        other_id = b_id if rows[index]["expected"] == "A" else a_id
        parsed = study.parse_choice(text)
        results.append(
            {
                "text": text,
                "generated_ids": ids,
                "output_token_count": effective,
                "parsed_choice": parsed,
                "raw_a_logit": float(raw[index, a_id]),
                "raw_b_logit": float(raw[index, b_id]),
                "processed_a_logit": float(processed[index, a_id]),
                "processed_b_logit": float(processed[index, b_id]),
                "decision_margin": float(raw[index, expected_id] - raw[index, other_id]),
                "first_generated_id": int(ids[0]),
                "first_raw_argmax_id": int(raw[index].argmax()),
                "a_token_id": int(a_id),
                "b_token_id": int(b_id),
                "technical_problem": None if parsed is not None else (
                    "output_truncation_preventing_scoring" if effective >= study.MAX_NEW_TOKENS else "parser_failure"
                ),
            }
        )
    return results


def evaluate_fixture(model: Any, tokenizer: Any, rows: list[dict[str, Any]], frozen: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = [study.chat_prompt(tokenizer, row["messages"]) for row in rows]
    final: list[dict[str, Any] | None] = [None] * len(rows)
    last_output: list[dict[str, Any] | None] = [None] * len(rows)
    attempt_logs: list[list[dict[str, Any]]] = [[] for _ in rows]
    invocation_errors = []
    for attempt in range(1, 2 + study.TECHNICAL_RETRIES):
        pending = [index for index, item in enumerate(final) if item is None]
        if not pending:
            break
        try:
            outputs = _decode_attempt(
                model,
                tokenizer,
                [rows[index] for index in pending],
                [prompts[index] for index in pending],
            )
            invocation_error = None
        except Exception as error:
            outputs = [None] * len(pending)
            invocation_error = f"{type(error).__name__}: {error}"
            invocation_errors.append({"attempt": attempt, "error": invocation_error})
        for index, output in zip(pending, outputs):
            if output is None:
                attempt_logs[index].append({"attempt": attempt, "invocation_error": invocation_error})
                continue
            last_output[index] = output
            attempt_logs[index].append(
                {
                    "attempt": attempt,
                    "text": output["text"],
                    "generated_ids": output["generated_ids"],
                    "technical_problem": output["technical_problem"],
                }
            )
            if output["technical_problem"] is None:
                final[index] = output
    records = []
    for index, (row, output) in enumerate(zip(rows, final)):
        base = {
            "observation_id": row["observation_id"],
            "fixture_id": row["fixture_id"],
            "condition": row["condition"],
            "rendering_hash": row["rendering_hash"],
            "messages_sha256": row["messages_sha256"],
            "chat_prompt_sha256": row["chat_prompt_sha256"],
            "model": study.MODEL_ID,
            "revision": study.MODEL_REVISION,
            "inference_config_hash": frozen["inference_config_hash"],
            "scorer_config_hash": frozen["scorer_hash"],
            "attempt_count": len(attempt_logs[index]),
            "attempt_log": attempt_logs[index],
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            "commit": os.environ.get("GITHUB_SHA", "local"),
            "completed_at": now(),
        }
        if output is None:
            retained = last_output[index]
            records.append(
                {
                    **base,
                    **(retained or {}),
                    "technical_status": "TECHNICAL_INVALID",
                    "technical_failure_reason": "parser/model invocation failure after frozen retries",
                    "invocation_errors": invocation_errors,
                    "parsed_choice": None,
                    "passed": None,
                    "expected": row["expected"],
                    "seed": study.SEED,
                    "sampling": {"do_sample": False, "max_new_tokens": study.MAX_NEW_TOKENS},
                }
            )
        else:
            records.append(
                {
                    **base,
                    **output,
                    "technical_status": "VALID",
                    "technical_failure_reason": None,
                    "passed": output["parsed_choice"] == row["expected"],
                    "expected": row["expected"],
                    "seed": study.SEED,
                    "sampling": {"do_sample": False, "max_new_tokens": study.MAX_NEW_TOKENS},
                }
            )
    return records


def run_shard(index: int, freeze_hash: str) -> None:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not 0 <= index < shard_count():
        raise ValueError("Invalid shard index")
    existing = load_shard(index, freeze_hash)
    if existing and existing["status"] == "COMPLETE":
        print(f"Shard {index} already complete; all frozen bindings match, so no observation is repeated.")
        return
    torch.manual_seed(study.SEED)
    np.random.seed(study.SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(study.MODEL_ID, revision=study.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        study.MODEL_ID, revision=study.MODEL_REVISION, torch_dtype=torch.bfloat16
    )
    model.eval()
    frozen = manifest()
    rows = rows_for_shard(index)
    by_fixture: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_fixture.setdefault(row["fixture_id"], []).append(row)
    fixture_ids = sorted(by_fixture, key=lambda fid: min(row["evaluation_order"] for row in by_fixture[fid]))
    payload = existing or {
        "protocol": study.VERSION,
        "freeze_manifest_hash": freeze_hash,
        "model": study.MODEL_ID,
        "revision": study.MODEL_REVISION,
        "inference_config_hash": frozen["inference_config_hash"],
        "scorer_config_hash": frozen["scorer_hash"],
        "shard_index": index,
        "status": "PARTIAL",
        "completed_fixture_ids": [],
        "records": [],
        "execution_attempts": [],
    }
    payload["execution_attempts"].append(
        {
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            "started_at": now(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        }
    )
    RESULTS.joinpath("shards").mkdir(parents=True, exist_ok=True)
    study.atomic_json(shard_path(index), payload)
    started = time.monotonic()
    soft_seconds = int(os.environ.get("RES_SFE_SOFT_SECONDS", "14400"))
    for fixture_id in fixture_ids[len(payload["completed_fixture_ids"]) :]:
        if time.monotonic() - started >= soft_seconds:
            print("Soft wall-clock ceiling reached before next fixture; checkpoint is resumable.")
            break
        fixture_rows = sorted(by_fixture[fixture_id], key=lambda row: row["condition_position"])
        records = evaluate_fixture(model, tokenizer, fixture_rows, frozen)
        payload["records"].extend(records)
        payload["completed_fixture_ids"].append(fixture_id)
        payload["status"] = "COMPLETE" if len(payload["completed_fixture_ids"]) == study.SHARD_FIXTURES else "PARTIAL"
        payload["updated_at"] = now()
        validate_shard(payload, index, freeze_hash)
        study.atomic_json(shard_path(index), payload)
        print(
            f"Shard {index}: checkpointed fixture {len(payload['completed_fixture_ids'])}/{study.SHARD_FIXTURES} "
            f"({len(payload['records'])} observations).",
            flush=True,
        )
        if time.monotonic() - started >= soft_seconds:
            print("Soft wall-clock ceiling reached after completing the current fixture.")
            break


def _choose_checkpoint(candidates: list[dict[str, Any]], existing: dict[str, Any] | None, index: int, freeze_hash: str) -> dict[str, Any] | None:
    valid = []
    if existing:
        valid.append(existing)
    for candidate in candidates:
        validate_shard(candidate, index, freeze_hash)
        valid.append(candidate)
    if not valid:
        return None
    valid.sort(key=lambda payload: len(payload["completed_fixture_ids"]))
    for shorter, longer in zip(valid, valid[1:]):
        if longer["completed_fixture_ids"][: len(shorter["completed_fixture_ids"])] != shorter["completed_fixture_ids"]:
            raise ValueError("Competing shard checkpoints disagree on fixture prefix")
        short_records = {record["observation_id"]: record for record in shorter["records"]}
        long_records = {record["observation_id"]: record for record in longer["records"]}
        if any(long_records.get(key) != value for key, value in short_records.items()):
            raise ValueError("Refusing to overwrite a prior frozen observation")
    return valid[-1]


def collect(artifact_root: Path, freeze_hash: str) -> dict[str, Any]:
    RESULTS.joinpath("shards").mkdir(parents=True, exist_ok=True)
    shard_states = {}
    all_records = []
    for index in range(shard_count()):
        existing = load_shard(index, freeze_hash)
        candidates = []
        for path in artifact_root.rglob(f"shard{index:02d}.json"):
            candidates.append(json.loads(path.read_text(encoding="utf-8")))
        chosen = _choose_checkpoint(candidates, existing, index, freeze_hash)
        if chosen:
            study.atomic_json(shard_path(index), chosen)
            all_records.extend(chosen["records"])
            shard_states[str(index)] = {
                "status": chosen["status"],
                "completed_fixtures": len(chosen["completed_fixture_ids"]),
            }
        else:
            shard_states[str(index)] = {"status": "MISSING", "completed_fixtures": 0}
    observation_ids = [record["observation_id"] for record in all_records]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("Duplicate observations across collected shards")
    all_records.sort(key=lambda record: next(
        row["evaluation_order"] for row in renderings() if row["observation_id"] == record["observation_id"]
    ))
    study.write_jsonl(RESULTS / "observations.jsonl", all_records)
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in artifact_root.rglob("token_audit.json")]
    if audits:
        first = audits[0]
        if any(audit != first for audit in audits[1:]) or first["freeze_manifest_hash"] != freeze_hash:
            raise ValueError("Tokenizer audits disagree or do not match freeze")
        study.atomic_json(RESULTS / "token_audit.json", first)
    completed_fixtures = sum(value["completed_fixtures"] for value in shard_states.values())
    expected_fixtures = len(fixtures())
    status = "COMPLETE_ATTEMPTED_ALL_FIXTURES" if completed_fixtures == expected_fixtures else "INCOMPLETE_RESUMABLE"
    payload = {
        "status": status,
        "protocol": study.VERSION,
        "freeze_manifest_hash": freeze_hash,
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "expected_fixtures": expected_fixtures,
        "completed_fixtures": completed_fixtures,
        "expected_observations": expected_fixtures * len(study.CONDITIONS),
        "stored_observations": len(all_records),
        "shards": shard_states,
    }
    study.atomic_json(RESULTS / "run_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "token-audit", "run", "collect"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--artifacts", type=Path)
    arguments = parser.parse_args()
    freeze_hash = verify_hashes()
    if arguments.command == "preflight":
        print(json.dumps({"status": "PHASE0_FROZEN", "freeze_manifest_hash": freeze_hash, "shards": shard_count()}))
    elif arguments.command == "token-audit":
        result = token_audit(freeze_hash)
        print(json.dumps({"status": "TOKEN_AUDIT_COMPLETE", "rows": len(result["rows"])}))
    elif arguments.command == "run":
        if arguments.shard is None:
            parser.error("--shard is required")
        run_shard(arguments.shard, freeze_hash)
    else:
        if arguments.artifacts is None:
            parser.error("--artifacts is required")
        result = collect(arguments.artifacts, freeze_hash)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
