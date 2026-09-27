#!/usr/bin/env python3
"""Condition-blind deterministic validator for rendered relation graphs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CANONICAL_EDGES = (
    ("REQUESTER", "COORDINATOR"),
    ("COORDINATOR", "EXECUTION_ACTOR"),
    ("REFERENCE_ACTOR", "OBSERVER"),
    ("OBSERVER", "REQUESTER"),
)


def parse_packet(packet: dict[str, Any]) -> dict[str, Any]:
    text = packet["text"]
    if "ROLE DEFINITIONS:" in text:
        surface_form = "VERBOSE"
        required = (
            "REQUESTER directs COORDINATOR to delegate the action to EXECUTION ACTOR",
            "COORDINATOR delegates execution to EXECUTION ACTOR",
            "REFERENCE ACTOR supplies a reference record to OBSERVER",
            "OBSERVER relays that record to REQUESTER",
            "EXECUTION ACTOR:",
            "REFERENCE ACTOR:",
        )
        valid_path = all(fragment in text for fragment in required)
        locus_match = re.search(r"DECISION LOCUS: (EXECUTION ACTOR|REFERENCE ACTOR)\.", text)
        marker_match = re.search(r"DOCUMENT MARKER: ([A-Z])\.", text)
        locus = locus_match.group(1).replace(" ", "_") if locus_match else None
    elif "ROLE KEY:" in text:
        surface_form = "ABBREVIATED"
        valid_path = (
            "RQ=requester" in text
            and "CO=coordinator" in text
            and "EX=execution actor" in text
            and "RF=reference actor" in text
            and "OB=observer" in text
            and "PATH: RQ>CO>EX; RF>OB>RQ." in text
            and "\n\nEX:" in text
            and "\n\nRF:" in text
        )
        locus_match = re.search(r"LOCUS: (EX|RF)\.", text)
        marker_match = re.search(r"MARKER: ([A-Z])\.", text)
        locus = {"EX": "EXECUTION_ACTOR", "RF": "REFERENCE_ACTOR"}.get(
            locus_match.group(1) if locus_match else ""
        )
    elif "KEY: R=requester" in text:
        surface_form = "SYMBOLIC"
        valid_path = (
            "C=coordinator" in text
            and "X=execution actor" in text
            and "Y=reference actor" in text
            and "O=observer" in text
            and "G:R>C>X;Y>O>R." in text
            and "\n\nX:" in text
            and "\n\nY:" in text
        )
        locus_match = re.search(r"(?:^|\n)L:([XY])\.", text)
        marker_match = re.search(r"(?:^|\n)M:([XY])\.", text)
        locus = {"X": "EXECUTION_ACTOR", "Y": "REFERENCE_ACTOR"}.get(
            locus_match.group(1) if locus_match else ""
        )
    else:
        surface_form = "UNKNOWN"
        valid_path = False
        locus = None
        marker_match = None
    marker = marker_match.group(1) if marker_match else None
    valid = bool(valid_path and locus and marker)
    return {
        "packet_id": packet["packet_id"],
        "valid": valid,
        "surface_form": surface_form,
        "graph_signature": {
            "edges": [list(edge) for edge in CANONICAL_EDGES],
            "decision_locus": locus,
        } if valid else None,
        "marker": marker,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packets", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    packets = [json.loads(line) for line in arguments.packets.read_text(encoding="utf-8").splitlines() if line]
    judgments = [parse_packet(packet) for packet in packets]
    arguments.output.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in judgments),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
