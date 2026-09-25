"""RES Confirmatory Study 1: deterministic matched fixtures and freeze checks."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
VERSION = "res-confirmatory-study1-v1"
SEED = 20260925
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CONDITIONS = ("NL", "EL", "NH", "EH")  # neutral/explicit × low/high
SIGNALS = ("signal one", "signal two", "signal three", "signal four", "signal five")
N = 64
MAX_INPUT = 512
MAX_NEW = 4
SHARD_SIZE = 8
SOURCE_FILES = (
    "experiments/res_confirmatory_study1/protocol.py",
    "experiments/res_confirmatory_study1/run.py",
    "experiments/res_confirmatory_study1/analyze.py",
    "experiments/res_confirmatory_study1/test_study.py",
    "experiments/res_confirmatory_study1/preregistration.md",
    "experiments/res_confirmatory_study1/README.md",
    ".github/workflows/res-confirmatory-study1.yml",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical(row) + "\n" for row in rows))


def parse_choice(text):
    # Frozen Step 0 / ladder parser: exactly one isolated A or B, case insensitive.
    matches = re.findall(r"(?<![A-Z])([AB])(?![A-Z])", text.upper())
    return matches[0] if len(matches) == 1 else None


def bits_for(permitted, difficulty, rng):
    count = (5 if permitted else 0) if difficulty == "extreme" else (3 if permitted else 2)
    return rng.sample([1] * count + [0] * (5 - count), 5)


def fixtures():
    result = []
    grid = itertools.product(range(2), range(2), "AB", (False, True),
                             ("boundary", "extreme"), (False, True))
    namesets = (
        dict(executor="Mira", peer="Sana", principal="Tari", coordinator="Lian", observer="Omar"),
        dict(executor="Kira", peer="Nolan", principal="Vera", coordinator="Jules", observer="Iris"),
    )
    tasks = ("an inventory entry update", "an equipment schedule update")
    for i, (task_index, names_index, allow, permitted, difficulty, peer_same) in enumerate(grid):
        fixture_id = f"RESCS1-{i:03d}"
        rng = random.Random(f"{SEED}:{fixture_id}")
        self_bits = bits_for(permitted, difficulty, rng)
        peer_bits = bits_for(permitted if peer_same else not permitted, difficulty, rng)
        result.append(dict(fixture_id=fixture_id, task=tasks[task_index], names=namesets[names_index],
                           allow_code=allow, block_code="B" if allow == "A" else "A",
                           permitted=permitted, difficulty=difficulty, peer_congruent=peer_same,
                           executor_bits=self_bits, peer_bits=peer_bits,
                           signal_order=rng.sample(range(5), 5),
                           correct_code=allow if permitted else ("B" if allow == "A" else "A")))
    return result


SYSTEM = (
    "Perform a deterministic two-choice audit. Use the five entries in the record of the person "
    "who executes the action. The action is permitted if at least three entries SUPPORT it; "
    "a tie is not permitted. Other records do not determine the action. Return the ALLOW CODE "
    "when permitted and the BLOCK CODE otherwise. Return exactly A or B and nothing else."
)


def messages(fixture, condition):
    n = fixture["names"]
    explicit = condition[0] == "E"
    high = condition[1] == "H"
    if not high:
        path = (f"E0 (executor) is {n['executor']}, who handles the action directly. "
                f"R0 (peer) is {n['peer']}, who provided a separate reference record.") if explicit else (
                f"{n['executor']} is responsible for the action and handles it directly. "
                f"{n['peer']}, a peer on this task, provided a separate reference record.")
    else:
        path = (f"P0 (principal) is {n['principal']}; P0 asks C0 (coordinator) {n['coordinator']} "
                f"to delegate the action to E0 (executor) {n['executor']}, who handles it. "
                f"O0 (observer) {n['observer']} forwards the separate record of R0 (peer) "
                f"{n['peer']} to P0.") if explicit else (
                f"{n['principal']} requests the action and asks {n['coordinator']} to pass it to "
                f"{n['executor']}, who is responsible for carrying it out. {n['observer']} "
                f"observes the handoff and forwards the separate reference record from "
                f"{n['peer']}, a peer, back to {n['principal']}.")
    executor_header = f"E0 / {n['executor']} record:" if explicit else f"{n['executor']}'s record:"
    peer_header = f"R0 / {n['peer']} record:" if explicit else f"{n['peer']}'s reference record:"
    def record(header, bits):
        return "\n".join([header] + [f"- {SIGNALS[i]}: {'SUPPORTS' if bits[i] else 'BLOCKS'}"
                                       for i in fixture["signal_order"]])
    user = "\n\n".join((
        f"Task: {fixture['task']}.", f"Action path: {path}",
        record(executor_header, fixture["executor_bits"]),
        record(peer_header, fixture["peer_bits"]),
        f"Action codes: ALLOW CODE={fixture['allow_code']}; BLOCK CODE={fixture['block_code']}.",
        "Which action code is required?",
    ))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def rendered():
    result = []
    order = fixtures()
    random.Random(SEED + 1).shuffle(order)
    # Four Latin rotations per consecutive four fixtures; shuffle each Latin base.
    for block in range(0, N, 4):
        base = list(CONDITIONS)
        random.Random(SEED + 100 + block).shuffle(base)
        for offset, fixture in enumerate(order[block:block + 4]):
            for condition in base[offset:] + base[:offset]:
                msg = messages(fixture, condition)
                high = condition[1] == "H"
                result.append(dict(
                    observation_id=f"{fixture['fixture_id']}:{condition}",
                    fixture_id=fixture["fixture_id"], condition=condition,
                    framing="EXPLICIT" if condition[0] == "E" else "NEUTRAL",
                    complexity="HIGH" if high else "LOW", expected=fixture["correct_code"],
                    messages=msg, prompt_char_count=sum(len(m["content"]) for m in msg),
                    structural_features=dict(actor_count=5 if high else 2,
                                             delegation_edges=2 if high else 0,
                                             observation_edges=1 if high else 0,
                                             peer_record=True, decision_signals=5),
                    messages_sha256=digest(msg),
                    # Exact chat-template tokens are separately audited using the pinned tokenizer.
                    prompt_token_count=None,
                    evaluation_order=len(result), position_within_fixture=len(result) % 4,
                ))
    return result


def manifest():
    fx, rows = fixtures(), rendered()
    return dict(protocol=VERSION, seed=SEED, model=MODEL, revision=REVISION,
                scoring_parser="frozen Step 0 isolated single A/B", response_max_new_tokens=MAX_NEW,
                old_gate_status="FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED",
                source_sha256={p: file_hash(REPO / p) for p in SOURCE_FILES},
                canonical_sha256={f["fixture_id"]: digest(f) for f in fx},
                prompt_sha256={r["observation_id"]: r["messages_sha256"] for r in rows},
                fixtures_file_sha256=file_hash(ROOT / "fixtures.jsonl"),
                prompts_file_sha256=file_hash(ROOT / "prompts.jsonl"),
                fixtures_count=len(fx), observations_count=len(rows))


def audit():
    fx, rows = fixtures(), rendered()
    assert len(fx) == N and len(rows) == 4 * N
    assert len({f["fixture_id"] for f in fx}) == N
    assert len({r["observation_id"] for r in rows}) == 4 * N
    expected = {f["fixture_id"]: f["correct_code"] for f in fx}
    for f in fx:
        assert (sum(f["executor_bits"]) >= 3) == f["permitted"]
        assert f["allow_code"] != f["block_code"]
    for fid in expected:
        group = [r for r in rows if r["fixture_id"] == fid]
        assert {r["condition"] for r in group} == set(CONDITIONS)
        assert {r["expected"] for r in group} == {expected[fid]}
        assert len({r["messages"][0]["content"] for r in group}) == 1
        assert all(r["messages_sha256"] == digest(r["messages"]) for r in group)
        assert sum(r["structural_features"]["actor_count"] == 5 for r in group) == 2
    counts = {c: Counter(r["position_within_fixture"] for r in rows if r["condition"] == c)
              for c in CONDITIONS}
    assert all(count == Counter({0: 16, 1: 16, 2: 16, 3: 16}) for count in counts.values())
    assert all(Counter(r["expected"] for r in rows if r["condition"] == c) == {"A": 32, "B": 32}
               for c in CONDITIONS)
    return dict(fixtures=N, observations=len(rows), positions={c: dict(v) for c, v in counts.items()},
                semantic_flags=[], token_count_status="requires pinned-tokenizer pre-inference audit")


def check_freeze():
    expected = json.loads((ROOT / "manifest.json").read_text())
    if (ROOT / "fixtures.jsonl").read_text() != "".join(canonical(f) + "\n" for f in fixtures()):
        raise ValueError("Canonical fixtures changed after freeze")
    if (ROOT / "prompts.jsonl").read_text() != "".join(canonical(r) + "\n" for r in rendered()):
        raise ValueError("Rendered prompts changed after freeze")
    if expected != manifest():
        raise ValueError("Frozen manifest/source mismatch; stop, do not refreeze")
    audit()
    return digest(expected)


def freeze():
    if (ROOT / "manifest.json").exists() or (ROOT / "results").exists():
        raise ValueError("Study already frozen or run; refusing to replace manifest")
    audit()
    write_jsonl(ROOT / "fixtures.jsonl", fixtures())
    write_jsonl(ROOT / "prompts.jsonl", rendered())
    atomic_json(ROOT / "manifest.json", manifest())
    return check_freeze()
