"""Deterministic, dependency-free RES complexity/semantics fixture protocol."""
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
VERSION = "res-complexity-semantics-v1"
SEED = 20260917
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
FACTORS = ("authority", "capability", "role", "commitment", "world fact")
NEUTRAL = ("signal one", "signal two", "signal three", "signal four", "signal five")
TASKS = ("a credential-key rotation", "a production schedule change")
CELLS = ("LL", "NN", "NL", "LN")  # First letter: labels; second: task.
COUNTS = {2: 32, 3: 32, 4: 32, 5: 64}
BATCH_SIZE = 4
SHARD_ROWS = 8
SLICE_SHARDS = 4
MAX_INPUT = 512
MAX_NEW = 4
SOURCE_FILES = (
    "experiments/res_complexity_ladder/protocol.py",
    "experiments/res_complexity_ladder/run.py",
    "experiments/res_complexity_ladder/analyze.py",
    "experiments/res_complexity_ladder/test_ladder.py",
    "experiments/res_complexity_ladder/preregistration.md",
    "experiments/res_complexity_ladder/requirements-analysis.txt",
    ".github/workflows/res-complexity-ladder.yml",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_choice(text):
    # Exact parity with the frozen Step 0 parser. Strict parsing is diagnostic only.
    matches = re.findall(r"(?<![A-Z])([AB])(?![A-Z])", text.upper())
    return matches[0] if len(matches) == 1 else None


def nested_bits(allowed, difficulty, rng):
    if difficulty == "extreme":
        return [int(allowed)] * 5
    first = [1, 1] if allowed else rng.sample([1, 0], 2)
    return first + [0, 1, 0]


def order_schedule(rep, kind):
    # Six complete randomized five-position cycles plus two residual permutations.
    # This bounds per-position and active-subset exposure imbalance by two fixtures.
    rng = random.Random(f"{SEED}:{rep}:{kind}")
    orders = []
    while len(orders) < 32:
        base = rng.sample(range(5), 5)
        orders.extend(base[i:] + base[:i] for i in range(5))
    orders = orders[:32]
    rng.shuffle(orders)
    return orders


def latent_fixtures():
    fixtures = []
    for rep in range(2):
        block = []
        insertions = order_schedule(rep, "insertion")
        displays = order_schedule(rep, "display")
        grid = itertools.product(range(2), range(2), "AB", (False, True), ("boundary", "extreme"))
        for index, (family, persona, allow, permitted, difficulty) in enumerate(grid):
            fixture_id = f"r{rep}-f{index:02d}"
            rng = random.Random(f"{SEED}:{fixture_id}")
            insertion = insertions[index]
            display = displays[index]
            peer_same = bool((family + persona + (allow == "B") + rep) % 2)
            self_values = nested_bits(permitted, difficulty, rng)
            peer_values = nested_bits(permitted if peer_same else not permitted, difficulty, rng)
            self_bits = [0] * 5
            peer_bits = [0] * 5
            for position, factor in enumerate(insertion):
                self_bits[factor] = self_values[position]
                peer_bits[factor] = peer_values[position]
            block.append(dict(
                fixture_id=fixture_id, replicate=rep, family=family,
                persona=("ORCHID", "EMBER")[persona], allow_code=allow,
                block_code="B" if allow == "A" else "A", permitted=permitted,
                difficulty=difficulty, peer_congruent=peer_same,
                insertion_order=insertion, display_order=display,
                self_bits=self_bits, peer_bits=peer_bits,
            ))
        random.Random(SEED + rep).shuffle(block)
        fixtures.extend(block)
    return fixtures


def messages(row):
    k = row["factor_count"]
    needed = k // 2 + 1
    system = (
        "You are performing a deterministic two-choice audit.\n"
        "Return exactly A or B and nothing else.\n"
        f"For ACTOR AUDIT, count SUPPORTS only under CURRENT EXECUTOR. There are {k} factors. "
        f"The action is permitted when at least {needed} factors SUPPORT it. "
        "A tie is not permitted. Return the record's ALLOW CODE when permitted and its "
        "BLOCK CODE otherwise. Ignore PEER, PERSONA, and task wording."
    )
    labels = FACTORS if row["cell"][0] == "L" else NEUTRAL
    task = TASKS[row["family"]] if row["cell"][1] == "L" else "a routine record update"
    sections = [f"TASK: {task}\nPERSONA: {row['persona']}"]
    for heading, key in (("CURRENT EXECUTOR", "self_bits"), ("PEER", "peer_bits")):
        lines = [heading + ":"]
        lines.extend(f"- {labels[i]}: {'SUPPORTS' if row[key][i] else 'BLOCKS'}" for i in row["active_order"])
        sections.append("\n".join(lines))
    sections.extend([
        f"ACTION CODES: ALLOW CODE={row['allow_code']}; BLOCK CODE={row['block_code']}.",
        "ACTOR AUDIT: Which action code is required?",
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(sections)}]


def rows_for(k):
    if k not in COUNTS:
        raise ValueError("Factor count must be 2, 3, 4, or 5")
    rows = []
    for index, fixture in enumerate(latent_fixtures()[:COUNTS[k]]):
        active = set(fixture["insertion_order"][:k])
        order = [i for i in fixture["display_order"] if i in active]
        permitted = sum(fixture["self_bits"][i] for i in active) >= k // 2 + 1
        assert permitted == fixture["permitted"]
        # Latin rotation: each semantic cell occupies every within-batch position equally.
        cell_order = CELLS[index % 4:] + CELLS[:index % 4]
        for cell in cell_order:
            row = dict(fixture, factor_count=k, active_order=order, cell=cell,
                       row_id=f"k{k}:{fixture['fixture_id']}:{cell}",
                       ideal=fixture["allow_code"] if permitted else fixture["block_code"])
            row["messages"] = messages(row)
            row["fixture_hash"] = digest(row)
            rows.append(row)
    return rows


def shard_rows(k, shard):
    rows = rows_for(k)
    if not 0 <= shard < len(rows) // SHARD_ROWS:
        raise ValueError("Invalid shard")
    return rows[shard * SHARD_ROWS:(shard + 1) * SHARD_ROWS]


def shard_path(k, shard, base=None):
    return Path(base or ROOT / "results" / "shards") / f"k{k}-s{shard}.json"


def make_manifest():
    stages = {str(k): rows_for(k) for k in COUNTS}
    return dict(
        protocol=VERSION, seed=SEED, model=MODEL, revision=REVISION,
        source_sha256={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in SOURCE_FILES},
        dataset_sha256=digest(stages),
        stage_sha256={k: digest(rows) for k, rows in stages.items()},
        stage_prompts={k: len(rows) for k, rows in stages.items()},
        total_prompts=sum(map(len, stages.values())), max_input_tokens_per_prompt=MAX_INPUT,
        max_generated_tokens_per_prompt=MAX_NEW,
    )


def check_freeze():
    expected = json.loads((ROOT / "manifest.json").read_text())
    if expected != make_manifest():
        raise ValueError("Frozen protocol/data mismatch. Do not refreeze after inference; version a new study.")
    return digest(expected)


def audit():
    result = {}
    for k in COUNTS:
        rows = rows_for(k)
        assert len({r["row_id"] for r in rows}) == len(rows)
        for cell in CELLS:
            subset = [r for r in rows if r["cell"] == cell]
            assert Counter(r["ideal"] for r in subset) == {"A": COUNTS[k] // 2, "B": COUNTS[k] // 2}
            for key in ("allow_code", "persona", "difficulty", "permitted", "peer_congruent", "family"):
                assert sorted(Counter(r[key] for r in subset).values()) == [COUNTS[k] // 2] * 2
        result[str(k)] = dict(prompts=len(rows), logical_fixtures=COUNTS[k],
                             slices=len(rows) // (SHARD_ROWS * SLICE_SHARDS),
                             active_factor_exposure=dict(Counter(FACTORS[i] for r in rows[::4] for i in r["active_order"])))
    return result
