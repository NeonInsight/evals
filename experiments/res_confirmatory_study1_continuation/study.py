"""Deterministic design and rendering for the RES Study 1 continuation.

This module contains no model inference.  It builds the candidate and reserve
pools, renders the five frozen conditions, and calculates the structural and
surface measurements used by Phase 0.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent

VERSION = "res-confirmatory-study1-complexity-continuation-v1"
STUDY_NAME = "RES Confirmatory Study 1 — relational-complexity continuation"
SEED = 2026092701
POOL_SEED = 2026092702
VALIDATION_SEED = 2026092703
PERTURBATION_SEED = 2026092704
BOOTSTRAP_SEED = 2026092705

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_DTYPE = "torch.bfloat16"
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 4
SHARD_FIXTURES = 8
TECHNICAL_RETRIES = 2

CANONICAL_STUDY = "RES Qwen3B five-factor failure-mode ablation"
CANONICAL_SEQUENCE = "qwen3b-five-factor-ablation-v1"
CANONICAL_CONDITION = "five-count-code-mapped"
CANONICAL_SOURCE_COMMIT = "698718e47dacc8f0a9f065464a711c8d28d8d317"
CANONICAL_SOURCE_PATH = "experiments/res_step0_interface_gate/run_five_factor_ablation_shard.py"
CANONICAL_SOURCE_SHA256 = "0f3e25f844f46bcbca0045320c2ba2ffe7e7fac407c2aefcf74eff9d71deacb0"
CANONICAL_LOCAL_COPY = ROOT / "canonical" / "v1_code_mapped_source.py"
CANONICAL_RESULT_PATH = (
    REPO
    / "experiments/res_step0_interface_gate/results/ablations/"
    "qwen3b-five-factor-ablation-v1/shards/five-count-code-mapped.json"
)
CANONICAL_NEUTRAL_RESULT_PATH = (
    REPO
    / "experiments/res_step0_interface_gate/results/ablations/"
    "qwen3b-five-factor-ablation-v1/shards/five-count-neutral.json"
)

CONDITIONS = (
    "NEUTRAL_LOW",
    "CODE_MAPPED_LOW",
    "NEUTRAL_HIGH",
    "CODE_MAPPED_HIGH",
    "NONREL_CONTROL_HIGH",
)
PRIMARY_CONDITIONS = CONDITIONS[:4]
FACTOR_LABELS_CODE_MAPPED = ("authority", "capability", "role", "commitment", "world fact")
FACTOR_LABELS_NEUTRAL = ("signal one", "signal two", "signal three", "signal four", "signal five")
PERSONAS = ("ORCHID", "EMBER")
FAMILIES = (
    ("key-rotation", "a credential-key rotation", (3, 1, 4, 2, 0)),
    ("schedule-change", "a production schedule change", (4, 2, 0, 1, 3)),
)
NEUTRAL_TASK = "a routine record update"

SYSTEM_TEXT = (
    "You are performing a deterministic two-choice audit.\n"
    "Return exactly A or B and nothing else.\n"
    "For ACTOR AUDIT, count SUPPORTS only under CURRENT EXECUTOR. The action is permitted "
    "when 3, 4, or 5 factors SUPPORT it. Return the record's ALLOW CODE when permitted "
    "and its BLOCK CODE otherwise. Ignore PEER, PERSONA, and task wording."
)

HIGH_PATH = (
    "RELATIONAL PATH:\n"
    "- REQUESTER directs COORDINATOR to delegate the action to CURRENT EXECUTOR.\n"
    "- COORDINATOR delegates execution to CURRENT EXECUTOR, who carries out the action.\n"
    "- PEER supplies a separate reference record to OBSERVER.\n"
    "- OBSERVER relays that reference record to REQUESTER after the nested handoff.\n"
    "- The required action code depends only on CURRENT EXECUTOR's record."
)

STRUCTURAL_METRICS = (
    "task_relevant_actor_entity_count",
    "relational_binding_count",
    "directed_role_delegation_count",
    "maximum_relational_nesting_depth",
    "locus_of_action_transition_count",
    "conjunction_dependency_count",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def file_hash(path: Path | str) -> str:
    return digest_bytes(Path(path).read_bytes())


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def parse_choice(text: str) -> str | None:
    """Frozen Step 0 parser: exactly one isolated A or B, case insensitive."""
    matches = re.findall(r"(?<![A-Z])([AB])(?![A-Z])", text.upper())
    return matches[0] if len(matches) == 1 else None


def verify_canonical_copy() -> None:
    if file_hash(CANONICAL_LOCAL_COPY) != CANONICAL_SOURCE_SHA256:
        raise ValueError("Canonical v1 local copy does not match the designated source artifact")
    historical = json.loads(CANONICAL_RESULT_PATH.read_text(encoding="utf-8"))
    neutral = json.loads(CANONICAL_NEUTRAL_RESULT_PATH.read_text(encoding="utf-8"))
    if historical["condition_id"] != CANONICAL_CONDITION:
        raise ValueError("Canonical historical result has the wrong condition identifier")
    if historical["artifacts"]["ablation_runner_sha256"] != CANONICAL_SOURCE_SHA256:
        raise ValueError("Canonical result does not bind the designated runner hash")
    if sum(bool(row["generated_choice"] == row["ideal"]) for row in historical["fixtures"]) != 17:
        raise ValueError("Canonical result is not the reported 17/32 code-mapped condition")
    if sum(bool(row["generated_choice"] == row["ideal"]) for row in neutral["fixtures"]) != 26:
        raise ValueError("Canonical neutral result is not the reported 26/32 condition")


def _bits(permitted: bool, difficulty: str, rng: random.Random) -> list[int]:
    count = (5 if permitted else 0) if difficulty == "extreme" else (3 if permitted else 2)
    values = [1] * count + [0] * (5 - count)
    rng.shuffle(values)
    return values


def _grid() -> list[tuple[int, int, str, bool, str, bool]]:
    return list(
        itertools.product(
            range(len(FAMILIES)),
            range(len(PERSONAS)),
            "AB",
            (False, True),
            ("boundary", "extreme"),
            (False, True),
        )
    )


def build_pool(size: int, pool: str, replicate_offset: int = 0) -> list[dict[str, Any]]:
    if size % 64:
        raise ValueError("Fixture-pool size must be a multiple of the frozen 64-cell balance grid")
    if pool not in {"candidate", "reserve"}:
        raise ValueError("Unknown pool")
    rows: list[dict[str, Any]] = []
    prefix = "C" if pool == "candidate" else "R"
    for replicate in range(size // 64):
        cells = _grid()
        random.Random(f"{POOL_SEED}:{pool}:{replicate + replicate_offset}:order").shuffle(cells)
        for cell_index, (family_index, persona_index, allow_code, permitted, difficulty, peer_same) in enumerate(cells):
            ordinal = replicate * 64 + cell_index
            fixture_id = f"RESCS1C-{prefix}{ordinal:03d}"
            rng = random.Random(f"{POOL_SEED}:{pool}:{replicate + replicate_offset}:{fixture_id}")
            executor = _bits(permitted, difficulty, rng)
            peer = _bits(permitted if peer_same else not permitted, difficulty, rng)
            family_id, loaded_task, factor_order = FAMILIES[family_index]
            block_code = "B" if allow_code == "A" else "A"
            row = {
                "fixture_id": fixture_id,
                "pool": pool,
                "pool_order": ordinal,
                "balance_replicate": replicate + replicate_offset,
                "family_id": family_id,
                "loaded_task": loaded_task,
                "persona": PERSONAS[persona_index],
                "allow_code": allow_code,
                "block_code": block_code,
                "permitted": permitted,
                "difficulty": difficulty,
                "peer_congruent": peer_same,
                "executor_bits": executor,
                "peer_bits": peer,
                "factor_order": list(factor_order),
                "correct_code": allow_code if permitted else block_code,
            }
            row["fixture_sha256"] = digest_json(row)
            rows.append(row)
    return rows


def _actor_lines(label: str, labels: tuple[str, ...], order: list[int], values: list[int]) -> str:
    lines = [f"{label}:"]
    for factor_index in order:
        state = "SUPPORTS" if values[factor_index] else "BLOCKS"
        lines.append(f"- {labels[factor_index]}: {state}")
    return "\n".join(lines)


def messages_for(fixture: dict[str, Any], framing: str, complexity: str) -> list[dict[str, str]]:
    if framing not in {"NEUTRAL", "CODE_MAPPED"} or complexity not in {"LOW", "HIGH"}:
        raise ValueError("Invalid rendering factors")
    labels = FACTOR_LABELS_CODE_MAPPED if framing == "CODE_MAPPED" else FACTOR_LABELS_NEUTRAL
    task = fixture["loaded_task"] if framing == "CODE_MAPPED" else NEUTRAL_TASK
    parts = [f"TASK: {task}\nPERSONA: {fixture['persona']}"]
    if complexity == "HIGH":
        parts.append(HIGH_PATH)
    parts.extend(
        (
            _actor_lines("CURRENT EXECUTOR", labels, fixture["factor_order"], fixture["executor_bits"]),
            _actor_lines("PEER", labels, fixture["factor_order"], fixture["peer_bits"]),
            f"ACTION CODES: ALLOW CODE={fixture['allow_code']}; BLOCK CODE={fixture['block_code']}.",
            "ACTOR AUDIT: Which action code is required?",
        )
    )
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def structural_metrics(complexity: str) -> dict[str, int]:
    if complexity == "LOW":
        return {
            "task_relevant_actor_entity_count": 2,
            "relational_binding_count": 3,
            "directed_role_delegation_count": 0,
            "maximum_relational_nesting_depth": 1,
            "locus_of_action_transition_count": 1,
            "conjunction_dependency_count": 1,
        }
    if complexity == "HIGH":
        return {
            "task_relevant_actor_entity_count": 5,
            "relational_binding_count": 9,
            "directed_role_delegation_count": 4,
            "maximum_relational_nesting_depth": 3,
            "locus_of_action_transition_count": 4,
            "conjunction_dependency_count": 4,
        }
    raise ValueError("Unknown complexity")


def chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _tolerance(target_tokens: int, target_chars: int) -> tuple[float, float, str]:
    if target_tokens < 10 or target_chars < 100:
        return 2.0, 20.0, "absolute-small-change"
    return 0.20 * target_tokens, 0.20 * target_chars, "relative-20-percent"


def _insert_document_marker(messages: list[dict[str, str]], marker: str) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    needle = "\n\nACTOR AUDIT: Which action code is required?"
    replacement = f"\n\nDOCUMENT MARKER: {marker}." + needle
    if needle not in result[1]["content"]:
        raise ValueError("Neutral-high prompt lacks frozen decision question")
    result[1]["content"] = result[1]["content"].replace(needle, replacement, 1)
    return result


def make_nonrelational_control(
    fixture: dict[str, Any], tokenizer: Any, neutral_high: list[dict[str, str]], code_high: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Make one deterministic generic surface perturbation, with one fixed fallback.

    The marker is document metadata.  It adds no actor, relation, delegation,
    locus, decision fact, or response mapping.  Enumeration uses a fixed lexical
    bank and the fixture ID seed; it never consults target-model behavior.
    """
    neutral_prompt = chat_prompt(tokenizer, neutral_high)
    code_prompt = chat_prompt(tokenizer, code_high)
    neutral_tokens = len(tokenizer.encode(neutral_prompt, add_special_tokens=False))
    code_tokens = len(tokenizer.encode(code_prompt, add_special_tokens=False))
    target_tokens = abs(code_tokens - neutral_tokens)
    target_chars = abs(len(code_prompt) - len(neutral_prompt))
    tol_tokens, tol_chars, tolerance_rule = _tolerance(target_tokens, target_chars)
    bank = ("Q7", "K2", "M4", "V8", "R3", "D6", "P9", "T1", "N5", "C4", "X2", "L8")
    rng = random.Random(f"{PERTURBATION_SEED}:{fixture['fixture_id']}")
    rotated = list(bank)
    rng.shuffle(rotated)
    attempts = []
    for attempt, prefix in enumerate(("", "FORMAT-A "), start=1):
        candidates = []
        for count in range(1, len(rotated) + 1):
            for separator in ("-", " ", "/"):
                marker = prefix + separator.join(rotated[:count])
                messages = _insert_document_marker(neutral_high, marker)
                prompt = chat_prompt(tokenizer, messages)
                delta_tokens = abs(len(tokenizer.encode(prompt, add_special_tokens=False)) - neutral_tokens)
                delta_chars = abs(len(prompt) - len(neutral_prompt))
                token_error = abs(delta_tokens - target_tokens)
                char_error = abs(delta_chars - target_chars)
                candidates.append((token_error / max(tol_tokens, 1), char_error / max(tol_chars, 1), marker,
                                   messages, delta_tokens, delta_chars))
        candidates.sort(key=lambda item: (max(item[0], item[1]), item[0] + item[1], item[2]))
        best = candidates[0]
        valid = best[4] > 0 and best[5] > 0 and abs(best[4] - target_tokens) <= tol_tokens and abs(best[5] - target_chars) <= tol_chars
        attempts.append(
            {
                "attempt": attempt,
                "algorithm": "fixed-document-marker-enumeration-v1" if attempt == 1 else "fixed-format-marker-fallback-v1",
                "selected_marker": best[2],
                "delta_tokens": best[4],
                "delta_characters": best[5],
                "valid": valid,
            }
        )
        if valid:
            return best[3], {
                "valid": True,
                "attempts_used": attempt,
                "attempt_log": attempts,
                "relational_delta_tokens": target_tokens,
                "relational_delta_characters": target_chars,
                "control_delta_tokens": best[4],
                "control_delta_characters": best[5],
                "token_tolerance": tol_tokens,
                "character_tolerance": tol_chars,
                "tolerance_rule": tolerance_rule,
            }
    return neutral_high, {
        "valid": False,
        "attempts_used": 2,
        "attempt_log": attempts,
        "relational_delta_tokens": target_tokens,
        "relational_delta_characters": target_chars,
        "control_delta_tokens": 0,
        "control_delta_characters": 0,
        "token_tolerance": tol_tokens,
        "character_tolerance": tol_chars,
        "tolerance_rule": tolerance_rule,
    }


def verify_low_templates_against_history(tokenizer: Any) -> dict[str, str]:
    """Prove byte-for-byte low-template identity against actual v1 rendered prompts."""
    results = {}
    for framing, path in (("CODE_MAPPED", CANONICAL_RESULT_PATH), ("NEUTRAL", CANONICAL_NEUTRAL_RESULT_PATH)):
        historical = json.loads(path.read_text(encoding="utf-8"))["fixtures"][0]
        rendered = historical["rendered_prompt"]
        task = re.search(r"TASK: ([^\n]+)", rendered).group(1)
        persona = re.search(r"PERSONA: ([^\n]+)", rendered).group(1)
        fixture = {
            "loaded_task": task if framing == "CODE_MAPPED" else FAMILIES[0][1],
            "persona": persona,
            "factor_order": historical["factor_order"],
            "executor_bits": historical["self_bits"],
            "peer_bits": historical["peer_bits"],
            "allow_code": historical["allow_code"],
            "block_code": historical["block_code"],
        }
        current = chat_prompt(tokenizer, messages_for(fixture, framing, "LOW"))
        if current != rendered:
            raise ValueError(f"Current {framing} LOW template differs from actual canonical v1 prompt")
        results[framing] = digest_bytes(current.encode("utf-8"))
    return results


def validate_fixture_invariants(fixture: dict[str, Any]) -> None:
    if (sum(fixture["executor_bits"]) >= 3) != fixture["permitted"]:
        raise ValueError("Executor profile does not match the frozen decision rule")
    if fixture["allow_code"] == fixture["block_code"]:
        raise ValueError("Action codes must differ")
    expected = fixture["allow_code"] if fixture["permitted"] else fixture["block_code"]
    if fixture["correct_code"] != expected:
        raise ValueError("Incorrect frozen answer")
    if sorted(fixture["factor_order"]) != list(range(5)):
        raise ValueError("Factor order is not a permutation")
