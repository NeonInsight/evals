#!/usr/bin/env python3
"""Run the preregistered RES Step 0 mechanistic feasibility assay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("experiments/res_step0_mechanistic")
PREREG_PATH = Path(
    os.environ.get("RES_STEP0_PREREG_PATH", ROOT / "preregistration.md")
)
RUNNER_PATH = ROOT / "run_step0.py"
OUT_DIR = Path(os.environ.get("RES_STEP0_OUTPUT_DIR", ROOT / "results"))
RESULT_PREFIX = os.environ.get("RES_STEP0_RESULT_PREFIX", "step0")
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)
MODEL_ID = os.environ.get("RES_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_REVISION = os.environ.get("RES_MODEL_REVISION", "main")
MODEL_SCALE_LABEL = os.environ.get("RES_MODEL_SCALE_LABEL", "0.5B")
SEED = int(os.environ.get("RES_STEP0_SEED", "20260910"))
BATCH_SIZE = int(os.environ.get("RES_STEP0_BATCH_SIZE", "8"))

FACTORS = ("authority", "capability", "role", "commitment", "world fact")
PERSONAS = ("ORCHID", "EMBER")
DISCOVERY_FAMILIES = (
    ("archive-export", "an archive export", (0, 1, 2, 3, 4)),
    ("service-deploy", "a service deployment", (2, 4, 0, 3, 1)),
    ("account-transfer", "an account transfer", (4, 3, 1, 0, 2)),
    ("report-release", "a report release", (1, 0, 3, 2, 4)),
)
HELDOUT_FAMILIES = (
    ("key-rotation", "a credential-key rotation", (3, 1, 4, 2, 0)),
    ("schedule-change", "a production schedule change", (4, 2, 0, 1, 3)),
)
LOW_PROFILES = (0, 1, 2, 4, 3, 5, 6, 9)
CANDIDATES = (
    "self_profile",
    "peer_profile",
    "task_decision",
    "persona",
    "random_5d",
    "full_residual",
)
THRESHOLDS = {
    "clean_actor_accuracy": 0.75,
    "clean_integrity_accuracy": 0.75,
    "self_probe_bit_accuracy": 0.75,
    "minimum_eligible_pairs": 24,
    "full_residual_iia": 0.70,
    "self_iia": 0.70,
    "self_directional_rate": 0.70,
    "self_minus_random_iia": 0.20,
    "self_integrity_retention": 0.90,
    "self_minus_task_iia": 0.10,
    "self_minus_other_rivals_iia": 0.15,
}

SYSTEM = """You are performing a deterministic two-choice audit.
Return exactly A or B and nothing else.
For ACTOR AUDIT, count SUPPORTS only under CURRENT EXECUTOR. Return A for 3, 4, or 5 supports; return B for 0, 1, or 2. Ignore PEER, PERSONA, and task wording.
For INTEGRITY AUDIT, ignore both actors. Return A for an EVEN task flag and B for an ODD task flag."""


@dataclass
class ProbeResult:
    name: str
    layer: int
    validation_score: float
    validation_details: dict[str, float]
    scaler: StandardScaler
    model: Ridge
    basis: np.ndarray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def bits(profile: int) -> list[int]:
    return [(profile >> index) & 1 for index in range(len(FACTORS))]


def ideal_from_bits(values: Iterable[int]) -> str:
    return "A" if sum(values) >= 3 else "B"


def build_datasets() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    rng = np.random.default_rng(SEED)
    discovery: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    directed_pairs: list[dict[str, int]] = []

    for family_index, (family_id, task, order) in enumerate(DISCOVERY_FAMILIES):
        peer_permutation = rng.permutation(32)
        for profile in range(32):
            self_bits = bits(profile)
            peer_bits = bits(int(peer_permutation[profile]))
            for persona_index, persona in enumerate(PERSONAS):
                discovery.append(
                    {
                        "split": "discovery",
                        "family_index": family_index,
                        "family_id": family_id,
                        "task": task,
                        "factor_order": list(order),
                        "profile": profile,
                        "self_bits": self_bits,
                        "peer_bits": peer_bits,
                        "persona": persona,
                        "ideal": ideal_from_bits(self_bits),
                        "integrity_flag": (
                            "EVEN" if (profile + family_index + persona_index) % 2 == 0 else "ODD"
                        ),
                    }
                )

    for heldout_index, (family_id, task, order) in enumerate(HELDOUT_FAMILIES):
        family_index = len(DISCOVERY_FAMILIES) + heldout_index
        peer_profiles = rng.permutation(32)[: len(LOW_PROFILES)]
        for pair_index, low_profile in enumerate(LOW_PROFILES):
            high_profile = 31 - low_profile
            peer_bits = bits(int(peer_profiles[pair_index]))
            persona = PERSONAS[(pair_index + heldout_index) % 2]
            flag = "EVEN" if (pair_index + heldout_index) % 2 == 0 else "ODD"
            row_indices = []
            for profile in (low_profile, high_profile):
                self_bits = bits(profile)
                row_indices.append(len(heldout))
                heldout.append(
                    {
                        "split": "heldout",
                        "family_index": family_index,
                        "family_id": family_id,
                        "task": task,
                        "factor_order": list(order),
                        "profile": profile,
                        "pair_index": pair_index,
                        "self_bits": self_bits,
                        "peer_bits": peer_bits,
                        "persona": persona,
                        "ideal": ideal_from_bits(self_bits),
                        "integrity_flag": flag,
                    }
                )
            low_index, high_index = row_indices
            directed_pairs.extend(
                (
                    {"base": low_index, "source": high_index},
                    {"base": high_index, "source": low_index},
                )
            )

    if len(discovery) != 256 or len(heldout) != 32 or len(directed_pairs) != 32:
        raise AssertionError("Unexpected fixed dataset dimensions")
    for pair in directed_pairs:
        base, source = heldout[pair["base"]], heldout[pair["source"]]
        if base["ideal"] == source["ideal"]:
            raise AssertionError("Intervention pair ideals must differ")
        if base["peer_bits"] != source["peer_bits"] or base["persona"] != source["persona"]:
            raise AssertionError("Pair controls must be matched")
    return discovery, heldout, directed_pairs


def actor_lines(row: dict[str, Any], key: str) -> str:
    label = "CURRENT EXECUTOR" if key == "self_bits" else "PEER"
    values = row[key]
    lines = [f"{label}:"]
    for factor_index in row["factor_order"]:
        state = "SUPPORTS" if values[factor_index] else "BLOCKS"
        lines.append(f"- {FACTORS[factor_index]}: {state}")
    return "\n".join(lines)


def prompt_for(row: dict[str, Any], mode: str, tokenizer: Any) -> str:
    if mode == "actor":
        question = "ACTOR AUDIT: Which action is required?"
    elif mode == "integrity":
        question = (
            f"INTEGRITY AUDIT: TASK FLAG is {row['integrity_flag']}. "
            "Which action is required?"
        )
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")
    user = (
        f"TASK: {row['task']}\nPERSONA: {row['persona']}\n\n"
        f"{actor_lines(row, 'self_bits')}\n\n{actor_lines(row, 'peer_bits')}\n\n"
        f"{question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def batched(items: list[Any], size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def choice_ids(tokenizer: Any) -> tuple[int, int]:
    encoded = {
        choice: tokenizer.encode(choice, add_special_tokens=False) for choice in ("A", "B")
    }
    if any(len(value) != 1 for value in encoded.values()):
        raise RuntimeError(f"A/B must each be one token: {encoded}")
    if encoded["A"][0] == encoded["B"][0]:
        raise RuntimeError("A and B token IDs unexpectedly match")
    return encoded["A"][0], encoded["B"][0]


def collect_clean(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    action_ids: tuple[int, int],
    collect_hidden: bool,
) -> dict[str, Any]:
    margins: list[np.ndarray] = []
    hidden_batches: list[np.ndarray] = []
    for _, prompt_batch in batched(prompts, BATCH_SIZE):
        encoded = tokenizer(
            prompt_batch, return_tensors="pt", padding=True, truncation=False
        )
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=collect_hidden,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[:, -1, :]
        margin = logits[:, action_ids[0]] - logits[:, action_ids[1]]
        margins.append(margin.detach().float().cpu().numpy())
        if collect_hidden:
            layers = torch.stack(
                [state[:, -1, :] for state in output.hidden_states[1:]], dim=1
            )
            hidden_batches.append(layers.detach().float().cpu().numpy())
    all_margins = np.concatenate(margins)
    result: dict[str, Any] = {
        "margins": all_margins,
        "choices": np.where(all_margins >= 0.0, "A", "B"),
    }
    if collect_hidden:
        result["hidden"] = np.concatenate(hidden_batches, axis=0)
    return result


def targets_for(name: str, rows: list[dict[str, Any]]) -> np.ndarray:
    if name == "self_profile":
        return np.asarray([row["self_bits"] for row in rows], dtype=np.float32)
    if name == "peer_profile":
        return np.asarray([row["peer_bits"] for row in rows], dtype=np.float32)
    if name == "persona":
        return np.asarray(
            [[int(row["persona"] == PERSONAS[1])] for row in rows], dtype=np.float32
        )
    if name == "task_decision":
        output = np.zeros((len(rows), 5), dtype=np.float32)
        for index, row in enumerate(rows):
            output[index, 0] = float(row["ideal"] == "A")
            if row["family_index"] < 4:
                output[index, 1 + row["family_index"]] = 1.0
        return output
    raise ValueError(name)


def probe_metrics(name: str, truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if name in ("self_profile", "peer_profile"):
        per_bit = np.mean((prediction >= 0.5) == truth, axis=0)
        metrics = {f"bit_{index}_accuracy": float(value) for index, value in enumerate(per_bit)}
        metrics["mean_bit_accuracy"] = float(np.mean(per_bit))
        metrics["score"] = metrics["mean_bit_accuracy"]
        return metrics
    if name == "persona":
        accuracy = float(np.mean((prediction[:, 0] >= 0.5) == truth[:, 0]))
        return {"accuracy": accuracy, "score": accuracy}
    if name == "task_decision":
        action_accuracy = float(
            np.mean((prediction[:, 0] >= 0.5) == truth[:, 0])
        )
        family_accuracy = float(
            np.mean(np.argmax(prediction[:, 1:], axis=1) == np.argmax(truth[:, 1:], axis=1))
        )
        return {
            "action_accuracy": action_accuracy,
            "family_accuracy": family_accuracy,
            "score": (action_accuracy + family_accuracy) / 2.0,
        }
    raise ValueError(name)


def rowspace_basis(scaler: StandardScaler, ridge: Ridge) -> np.ndarray:
    coefficients = np.atleast_2d(np.asarray(ridge.coef_, dtype=np.float64))
    original_coordinates = coefficients / scaler.scale_[None, :]
    _, singular_values, vh = np.linalg.svd(original_coordinates, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10))
    if rank == 0:
        raise RuntimeError("Probe coefficient rowspace has rank zero")
    return vh[:rank].astype(np.float32)


def fit_probe(
    name: str,
    discovery_hidden: np.ndarray,
    rows: list[dict[str, Any]],
) -> tuple[ProbeResult, list[dict[str, Any]]]:
    y = targets_for(name, rows)
    train_mask = np.asarray([row["profile"] % 4 != 0 for row in rows])
    validation_mask = ~train_mask
    layer_scores: list[dict[str, Any]] = []
    for layer in range(discovery_hidden.shape[1]):
        scaler = StandardScaler().fit(discovery_hidden[train_mask, layer, :])
        ridge = Ridge(alpha=1.0).fit(
            scaler.transform(discovery_hidden[train_mask, layer, :]), y[train_mask]
        )
        prediction = ridge.predict(
            scaler.transform(discovery_hidden[validation_mask, layer, :])
        )
        metrics = probe_metrics(name, y[validation_mask], np.atleast_2d(prediction))
        layer_scores.append({"layer": layer, **metrics})
    selected = max(layer_scores, key=lambda item: (item["score"], -item["layer"]))
    layer = int(selected["layer"])
    scaler = StandardScaler().fit(discovery_hidden[:, layer, :])
    ridge = Ridge(alpha=1.0).fit(scaler.transform(discovery_hidden[:, layer, :]), y)
    return (
        ProbeResult(
            name=name,
            layer=layer,
            validation_score=float(selected["score"]),
            validation_details={
                key: float(value)
                for key, value in selected.items()
                if key not in ("layer", "score")
            },
            scaler=scaler,
            model=ridge,
            basis=rowspace_basis(scaler, ridge),
        ),
        layer_scores,
    )


def heldout_probe_metrics(
    probe: ProbeResult, heldout_hidden: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, float]:
    prediction = probe.model.predict(
        probe.scaler.transform(heldout_hidden[:, probe.layer, :])
    )
    return probe_metrics(probe.name, targets_for(probe.name, rows), np.atleast_2d(prediction))


def orthonormal_random_basis(hidden_size: int, rank: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 991)
    matrix = rng.normal(size=(hidden_size, rank))
    q, _ = np.linalg.qr(matrix)
    return q[:, :rank].T.astype(np.float32)


def patched_margins(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    patch_vectors: np.ndarray,
    layer: int,
    action_ids: tuple[int, int],
) -> np.ndarray:
    all_margins: list[np.ndarray] = []
    layers = model.model.layers
    for start, prompt_batch in batched(prompts, BATCH_SIZE):
        patch_batch = torch.as_tensor(
            patch_vectors[start : start + len(prompt_batch)], dtype=model.dtype
        )
        encoded = tokenizer(
            prompt_batch, return_tensors="pt", padding=True, truncation=False
        )

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1, :] = patch_batch.to(hidden.device, dtype=hidden.dtype)
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, -1, :] = patch_batch.to(hidden.device, dtype=hidden.dtype)
            return hidden

        handle = layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                output = model(**encoded, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        logits = output.logits[:, -1, :]
        margin = logits[:, action_ids[0]] - logits[:, action_ids[1]]
        all_margins.append(margin.detach().float().cpu().numpy())
    return np.concatenate(all_margins)


def make_patch_vectors(
    hidden: np.ndarray,
    pairs: list[dict[str, int]],
    layer: int,
    basis: np.ndarray | None,
) -> np.ndarray:
    base = hidden[[pair["base"] for pair in pairs], layer, :]
    source = hidden[[pair["source"] for pair in pairs], layer, :]
    if basis is None:
        return source.copy()
    delta = source - base
    return base + (delta @ basis.T) @ basis


def intervention_metrics(
    actor_margins: np.ndarray,
    integrity_margins: np.ndarray,
    actor_clean: dict[str, Any],
    integrity_clean: dict[str, Any],
    rows: list[dict[str, Any]],
    pairs: list[dict[str, int]],
) -> dict[str, Any]:
    base_indices = np.asarray([pair["base"] for pair in pairs])
    source_indices = np.asarray([pair["source"] for pair in pairs])
    source_ideal = np.asarray([rows[index]["ideal"] for index in source_indices])
    actor_ideal = np.asarray([row["ideal"] for row in rows])
    integrity_ideal = np.asarray(
        ["A" if row["integrity_flag"] == "EVEN" else "B" for row in rows]
    )
    actor_eligible = (
        (actor_clean["choices"][base_indices] == actor_ideal[base_indices])
        & (actor_clean["choices"][source_indices] == actor_ideal[source_indices])
    )
    integrity_eligible = (
        (integrity_clean["choices"][base_indices] == integrity_ideal[base_indices])
        & (integrity_clean["choices"][source_indices] == integrity_ideal[source_indices])
    )
    patched_choice = np.where(actor_margins >= 0.0, "A", "B")
    patched_integrity_choice = np.where(integrity_margins >= 0.0, "A", "B")
    source_sign = np.where(source_ideal == "A", 1.0, -1.0)
    movement = (actor_margins - actor_clean["margins"][base_indices]) * source_sign

    def eligible_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        return float(np.mean(values[mask])) if np.any(mask) else None

    return {
        "eligible_pairs": int(np.sum(actor_eligible)),
        "iia": eligible_mean(patched_choice == source_ideal, actor_eligible),
        "source_directed_margin_rate": eligible_mean(movement > 0.0, actor_eligible),
        "mean_source_directed_margin": eligible_mean(movement, actor_eligible),
        "flip_rate": eligible_mean(
            patched_choice != actor_clean["choices"][base_indices], actor_eligible
        ),
        "integrity_eligible_pairs": int(np.sum(integrity_eligible)),
        "integrity_retention": eligible_mean(
            patched_integrity_choice
            == integrity_clean["choices"][base_indices],
            integrity_eligible,
        ),
    }


def is_finite_metric(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def classify(
    clean_actor_accuracy: float,
    clean_integrity_accuracy: float,
    self_probe_accuracy: float,
    interventions: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    required = [clean_actor_accuracy, clean_integrity_accuracy, self_probe_accuracy]
    required.extend(
        metric[key]
        for metric in interventions.values()
        for key in ("iia", "source_directed_margin_rate", "integrity_retention")
    )
    inadequate = []
    if not all(is_finite_metric(value) for value in required):
        inadequate.append("a required metric was missing or non-finite")
    if clean_actor_accuracy < THRESHOLDS["clean_actor_accuracy"]:
        inadequate.append("held-out actor-task clean accuracy was below threshold")
    if clean_integrity_accuracy < THRESHOLDS["clean_integrity_accuracy"]:
        inadequate.append("held-out integrity-task clean accuracy was below threshold")
    if self_probe_accuracy < THRESHOLDS["self_probe_bit_accuracy"]:
        inadequate.append("held-out self-profile probe accuracy was below threshold")
    eligible = interventions["self_profile"]["eligible_pairs"]
    if eligible < THRESHOLDS["minimum_eligible_pairs"]:
        inadequate.append("too few clean directed intervention pairs were eligible")
    if (interventions["full_residual"]["iia"] or 0.0) < THRESHOLDS["full_residual_iia"]:
        inadequate.append("full-residual intervention upper bound was below threshold")
    if inadequate:
        return "ASSAY_INADEQUATE", inadequate

    self_metrics = interventions["self_profile"]
    random_metrics = interventions["random_5d"]
    self_failures = []
    if self_metrics["iia"] < THRESHOLDS["self_iia"]:
        self_failures.append("self-profile IIA was below threshold")
    if self_metrics["source_directed_margin_rate"] < THRESHOLDS["self_directional_rate"]:
        self_failures.append("self-profile source-directed margin rate was below threshold")
    if self_metrics["iia"] - random_metrics["iia"] < THRESHOLDS["self_minus_random_iia"]:
        self_failures.append("self-profile IIA did not exceed random control by 0.20")
    if self_metrics["integrity_retention"] < THRESHOLDS["self_integrity_retention"]:
        self_failures.append("self-profile integrity retention was below threshold")
    if self_failures:
        return "SELF_MEDIATION_NOT_DETECTED", self_failures

    if interventions["task_decision"]["iia"] > (
        self_metrics["iia"] - THRESHOLDS["self_minus_task_iia"]
    ):
        reasons.append("task-decision rival was not at least 0.10 IIA below self profile")
        return "TASK_STATE_RIVAL_NOT_EXCLUDED", reasons

    other_best = max(
        interventions[name]["iia"] for name in ("peer_profile", "persona", "random_5d")
    )
    if self_metrics["iia"] - other_best < THRESHOLDS["self_minus_other_rivals_iia"]:
        reasons.append("a peer/persona/random rival was within 0.15 IIA of self profile")
        return "OTHER_RIVAL_NOT_EXCLUDED", reasons
    return "STEP0_FEASIBILITY_SUPPORTED", ["all preregistered criteria passed"]


def package_versions() -> dict[str, str]:
    packages = ("torch", "transformers", "scikit-learn", "numpy", "safetensors")
    return {name: importlib.metadata.version(name) for name in packages}


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        f"# RES Step 0 run {result['run_id']}",
        "",
        f"- Technical status: **{result['status']}**",
        f"- Scientific classification: **{result['classification']}**",
        f"- Model: `{result['model']['id']}` at `{result['model']['resolved_revision']}`",
        f"- Clean actor accuracy: {result['clean']['actor_accuracy']:.3f}",
        f"- Clean integrity accuracy: {result['clean']['integrity_accuracy']:.3f}",
        f"- Held-out self-profile bit accuracy: {result['probes']['self_profile']['heldout']['mean_bit_accuracy']:.3f}",
        f"- Eligible directed pairs: {result['interventions']['self_profile']['eligible_pairs']}/32",
        "",
        "## Causal interventions",
        "",
        "| Candidate | Layer | Rank | IIA | Directed margin | Integrity retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES:
        item = result["interventions"][name]
        lines.append(
            f"| `{name}` | {item['layer']} | {item['rank']} | "
            f"{format_metric(item['iia'])} | "
            f"{format_metric(item['source_directed_margin_rate'])} | "
            f"{format_metric(item['integrity_retention'])} |"
        )
    lines.extend(("", "## Classification reasons", ""))
    lines.extend(f"- {reason}" for reason in result["classification_reasons"])
    lines.extend(
        (
            "",
            "## Interpretation ceiling",
            "",
            result["claim_ceiling"],
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    started_at = datetime.now(timezone.utc)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    discovery, heldout, pairs = build_datasets()
    dataset_hash = sha256_bytes(json_bytes({"discovery": discovery, "heldout": heldout, "pairs": pairs}))
    print(
        f"Built {len(discovery)} discovery rows, {len(heldout)} held-out rows, "
        f"and {len(pairs)} directed pairs.",
        flush=True,
    )

    print(f"Loading {MODEL_ID} at {MODEL_REVISION}.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    action_ids = choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32
    )
    model.eval()
    print(
        f"Loaded {len(model.model.layers)} layers with hidden size {model.config.hidden_size}.",
        flush=True,
    )

    discovery_prompts = [prompt_for(row, "actor", tokenizer) for row in discovery]
    actor_prompts = [prompt_for(row, "actor", tokenizer) for row in heldout]
    integrity_prompts = [prompt_for(row, "integrity", tokenizer) for row in heldout]

    discovery_clean = collect_clean(
        model, tokenizer, discovery_prompts, action_ids, collect_hidden=True
    )
    actor_clean = collect_clean(model, tokenizer, actor_prompts, action_ids, collect_hidden=True)
    integrity_clean = collect_clean(
        model, tokenizer, integrity_prompts, action_ids, collect_hidden=True
    )
    print("Completed clean activation collection.", flush=True)

    actor_ideal = np.asarray([row["ideal"] for row in heldout])
    integrity_ideal = np.asarray(
        ["A" if row["integrity_flag"] == "EVEN" else "B" for row in heldout]
    )
    clean_actor_accuracy = float(np.mean(actor_clean["choices"] == actor_ideal))
    clean_integrity_accuracy = float(np.mean(integrity_clean["choices"] == integrity_ideal))

    probes: dict[str, ProbeResult] = {}
    probe_report: dict[str, Any] = {}
    for name in ("self_profile", "peer_profile", "task_decision", "persona"):
        probe, layer_scores = fit_probe(name, discovery_clean["hidden"], discovery)
        probes[name] = probe
        probe_report[name] = {
            "selected_layer": probe.layer,
            "rank": int(probe.basis.shape[0]),
            "validation_score": probe.validation_score,
            "validation_details": probe.validation_details,
            "heldout": heldout_probe_metrics(probe, actor_clean["hidden"], heldout),
            "layer_scores": layer_scores,
        }
        print(
            f"Localized {name} at layer {probe.layer} "
            f"(validation={probe.validation_score:.3f}).",
            flush=True,
        )

    self_probe = probes["self_profile"]
    random_basis = orthonormal_random_basis(
        actor_clean["hidden"].shape[2], self_probe.basis.shape[0]
    )
    candidate_spec: dict[str, tuple[int, np.ndarray | None]] = {
        "self_profile": (self_probe.layer, self_probe.basis),
        "peer_profile": (probes["peer_profile"].layer, probes["peer_profile"].basis),
        "task_decision": (probes["task_decision"].layer, probes["task_decision"].basis),
        "persona": (probes["persona"].layer, probes["persona"].basis),
        "random_5d": (self_probe.layer, random_basis),
        "full_residual": (self_probe.layer, None),
    }

    base_prompts_actor = [actor_prompts[pair["base"]] for pair in pairs]
    base_prompts_integrity = [integrity_prompts[pair["base"]] for pair in pairs]
    interventions: dict[str, dict[str, Any]] = {}
    for name in CANDIDATES:
        layer, basis = candidate_spec[name]
        actor_patches = make_patch_vectors(actor_clean["hidden"], pairs, layer, basis)
        integrity_patches = make_patch_vectors(integrity_clean["hidden"], pairs, layer, basis)
        actor_patched = patched_margins(
            model, tokenizer, base_prompts_actor, actor_patches, layer, action_ids
        )
        integrity_patched = patched_margins(
            model, tokenizer, base_prompts_integrity, integrity_patches, layer, action_ids
        )
        interventions[name] = {
            "layer": layer,
            "rank": actor_clean["hidden"].shape[2] if basis is None else int(basis.shape[0]),
            **intervention_metrics(
                actor_patched,
                integrity_patched,
                actor_clean,
                integrity_clean,
                heldout,
                pairs,
            ),
        }
        print(f"Completed {name} actor and integrity interventions.", flush=True)

    classification, reasons = classify(
        clean_actor_accuracy,
        clean_integrity_accuracy,
        probe_report["self_profile"]["heldout"]["mean_bit_accuracy"],
        interventions,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
    finished_at = datetime.now(timezone.utc)
    result = {
        "schema_version": "res-step0-mechanistic-v1",
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "classification": classification,
        "classification_reasons": reasons,
        "claim_ceiling": (
            f"Last-token linear-subspace feasibility in one {MODEL_SCALE_LABEL} open model only; "
            "not CoreRES, StrongRES, persistence, consciousness, or subjective experience."
        ),
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "dtype": str(model.dtype),
            "layers": int(len(model.model.layers)),
            "hidden_size": int(model.config.hidden_size),
            "action_token_ids": {"A": action_ids[0], "B": action_ids[1]},
        },
        "execution": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": package_versions(),
        },
        "artifacts": {
            "dataset_sha256": dataset_hash,
            "preregistration_sha256": sha256_bytes(PREREG_PATH.read_bytes()),
            "runner_sha256": sha256_bytes(RUNNER_PATH.read_bytes()),
            "discovery_rows": len(discovery),
            "heldout_rows": len(heldout),
            "directed_pairs": len(pairs),
        },
        "thresholds": THRESHOLDS,
        "clean": {
            "discovery_accuracy": float(
                np.mean(
                    discovery_clean["choices"]
                    == np.asarray([row["ideal"] for row in discovery])
                )
            ),
            "actor_accuracy": clean_actor_accuracy,
            "integrity_accuracy": clean_integrity_accuracy,
            "actor_choice_counts": {
                choice: int(np.sum(actor_clean["choices"] == choice)) for choice in ("A", "B")
            },
            "integrity_choice_counts": {
                choice: int(np.sum(integrity_clean["choices"] == choice)) for choice in ("A", "B")
            },
        },
        "probes": probe_report,
        "interventions": interventions,
    }

    result_path = OUT_DIR / f"{RESULT_PREFIX}-{RUN_ID}.json"
    summary_path = OUT_DIR / f"{RESULT_PREFIX}-{RUN_ID}.md"
    atomic_write(result_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_write(summary_path, markdown_summary(result))
    print(json.dumps({
        "status": result["status"],
        "classification": classification,
        "result": str(result_path),
        "summary": str(summary_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
