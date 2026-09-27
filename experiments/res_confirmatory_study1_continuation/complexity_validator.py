#!/usr/bin/env python3
"""Blinded deterministic relational-state validator.

The process accepts only randomized A/B text and the frozen question.  It has
no access to intended complexity, framing, hypotheses, answers, prior model
performance, or target-model observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

QUESTION = (
    "Which version requires maintaining more relational state, including actors, roles, "
    "delegation relationships, nested dependencies, or locus-of-action distinctions, "
    "in order to make the same underlying decision?"
)

ACTOR_MARKERS = ("CURRENT EXECUTOR", "PEER", "REQUESTER", "COORDINATOR", "OBSERVER")
DIRECTED_MARKERS = (
    "directs",
    "delegate",
    "delegates",
    "carries out",
    "supplies",
    "relays",
    "handoff",
)
DEPENDENCY_MARKERS = ("depends", "only", "after", "nested", "otherwise", "when")
LOCUS_MARKERS = ("CURRENT EXECUTOR", "carries out", "execution", "required action code", "record")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def features(text: str) -> dict[str, int]:
    upper = text.upper()
    lowered = text.lower()
    actors = sum(marker in upper for marker in ACTOR_MARKERS)
    directed = sum(len(re.findall(rf"\b{re.escape(marker)}\b", lowered)) for marker in DIRECTED_MARKERS)
    dependencies = sum(len(re.findall(rf"\b{re.escape(marker)}\b", lowered)) for marker in DEPENDENCY_MARKERS)
    loci = sum(len(re.findall(re.escape(marker.lower()), lowered)) for marker in LOCUS_MARKERS)
    nesting = 3 if "nested handoff" in lowered else (2 if "delegate" in lowered else 1)
    # Lexicographic ordering prevents prompt length from deciding the judgment.
    # Each component represents relational content; characters/tokens are absent.
    return {
        "actors": actors,
        "directed_relations": directed,
        "nesting": nesting,
        "locus_distinctions": loci,
        "dependencies": dependencies,
    }


def relational_tuple(values: dict[str, int]) -> tuple[int, ...]:
    return (
        values["actors"],
        values["directed_relations"],
        values["nesting"],
        values["locus_distinctions"],
        values["dependencies"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packets")
    parser.add_argument("output")
    arguments = parser.parse_args()
    packets = [json.loads(line) for line in Path(arguments.packets).read_text(encoding="utf-8").splitlines() if line]
    judgments = []
    for packet in packets:
        if set(packet) != {"packet_id", "question", "version_a", "version_b"}:
            raise ValueError("Blinded packet contains forbidden metadata")
        if packet["question"] != QUESTION:
            raise ValueError("Validation question differs from the frozen question")
        a, b = features(packet["version_a"]), features(packet["version_b"])
        score_a, score_b = relational_tuple(a), relational_tuple(b)
        choice = "A" if score_a > score_b else "B" if score_b > score_a else "TIE"
        judgments.append(
            {
                "packet_id": packet["packet_id"],
                "packet_sha256": sha(json.dumps(packet, sort_keys=True, separators=(",", ":"))),
                "choice": choice,
                "version_a_relational_features": a,
                "version_b_relational_features": b,
                "decision_rule": "lexicographic relational tuple; prompt length unavailable",
            }
        )
    Path(arguments.output).write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in judgments),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

