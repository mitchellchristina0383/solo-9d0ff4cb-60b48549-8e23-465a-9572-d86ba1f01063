"""Low-peak scheduler: metadata-only DP, exactness and executor agreement."""

from __future__ import annotations

import itertools
import time

import pytest

from memdry.memory import execute_plan
from memdry.model import Plan, TensorSpec, Op, ValidationError, parse_plan
from memdry.schedule import (
    MAX_SCHEDULE_OPS,
    build_schedule,
    describe_order,
    peak_for_order,
)


def make_plan(ops_text, inputs_text, outputs, capacity=100000):
    return parse_plan(
        {
            "capacity": str(capacity),
            "inputs": inputs_text,
            "ops": ops_text,
            "outputs": outputs,
        }
    )


def test_reorder_lowers_peak_on_hand_built_plan():
    # Original order produces the big 8-element relu first (peak 24); doing the
    # small dependent chain first lets B and C die before x is allocated.
    plan = make_plan(
        "x = relu(A)\ny = add(B, C)\nu = relu(y)",
        "A: 1, -2, 3, -4, 5, -6, 7, -8\n"
        "B: 1, 2, 3, 4\nC: 10, -10, 20, -20",
        "x, u",
    )
    schedule = build_schedule(plan)
    assert schedule.original_peak == 24
    assert schedule.min_elements == 20
    # Both (1, 2, 0) and (1, 0, 2) reach peak 20; the DP keeps the
    # lexicographically smallest optimal order.
    assert schedule.order in {(1, 2, 0), (1, 0, 2)}
    assert schedule.reordered is True
    assert schedule.saved == 4
    assert peak_for_order(plan, schedule.order) == 20

    # At capacity 21 the original order overflows while the selected order runs.
    tight = Plan(capacity=21, inputs=plan.inputs, ops=plan.ops, outputs=plan.outputs)
    assert execute_plan(tight)["ok"] is False
    reordered = execute_plan(tight, schedule.order)
    assert reordered["ok"] is True
    assert reordered["peak"]["elements"] == 20


def test_reordered_outputs_match_original_float32_values():
    plan = make_plan(
        "x = relu(A)\ny = add(B, C)\nu = relu(y)",
        "A: 1, -2, 3, -4, 5, -6, 7, -8\n"
        "B: 1, 2, 3, 4\nC: 10, -10, 20, -20",
        "x, u",
    )
    schedule = build_schedule(plan)
    original = execute_plan(plan)
    reordered = execute_plan(plan, schedule.order)
    assert original["ok"] and reordered["ok"]
    before = {o["name"]: o["values"] for o in original["outputs"]}
    after = {o["name"]: o["values"] for o in reordered["outputs"]}
    assert before == after
    assert after["x"] == ["1", "0", "3", "0", "5", "0", "7", "0"]
    assert after["u"] == ["11", "0", "23", "0"]


def test_slice_storage_is_counted_once_and_survives_source_release():
    plan = make_plan(
        "v = slice(A, 1, 4)\nr = relu(B)",
        "A: 10, -20, 30, -40, 50\nB: 1, 2",
        "v, r",
    )
    schedule = build_schedule(plan)
    # A(5) and B(2) loaded; v shares A's block, then relu allocates 2 while
    # A's block (kept alive by output view v) is still present: 5 + 2 + 2 = 9.
    assert schedule.min_elements == 9
    result = execute_plan(Plan(9, plan.inputs, plan.ops, plan.outputs), schedule.order)
    assert result["ok"]
    view = next(o for o in result["outputs"] if o["name"] == "v")
    assert view["role"] == "view"
    assert view["values"] == ["-20", "30", "-40"]


def test_original_order_is_kept_when_it_is_already_optimal():
    plan = make_plan(
        "s = add(a, b)\nr = relu(s)",
        "a: 1, 2, 3\nb: 4, 5, 6",
        "r",
    )
    schedule = build_schedule(plan)
    assert schedule.order == (0, 1)
    assert schedule.reordered is False
    assert schedule.saved == 0
    assert schedule.min_elements == schedule.original_peak


def test_more_than_twelve_ops_is_rejected():
    inputs = "x: 1, 2\n"
    ops = "\n".join(f"r{i} = relu({'x' if i == 0 else f'r{i-1}'})" for i in range(13))
    plan = parse_plan(
        {"capacity": "100", "inputs": inputs, "ops": ops, "outputs": "r12"}
    )
    assert len(plan.ops) == 13
    with pytest.raises(ValidationError) as info:
        build_schedule(plan)
    assert str(MAX_SCHEDULE_OPS) in info.value.messages[0]


def test_dp_matches_brute_force_on_all_small_random_plans():
    import random

    rng = random.Random(4242)
    for _ in range(40):
        n_ops = rng.randint(1, 6)
        inputs, lengths, known = [], {}, []
        for i in range(rng.randint(1, 4)):
            length = rng.randint(1, 4)
            name = f"in{i}"
            inputs.append(
                TensorSpec(name=name, values=tuple([1.0] * length), line=i + 1)
            )
            lengths[name] = length
            known.append(name)
        ops = []
        for i in range(n_ops):
            name = f"o{i}"
            kind = rng.choice(["add", "relu", "slice"])
            if kind == "relu":
                src = rng.choice(known)
                ops.append(Op(name, "relu", (src,), i + 1, length=lengths[src]))
            elif kind == "add":
                a = rng.choice(known)
                b = rng.choice([k for k in known if lengths[k] == lengths[a]])
                ops.append(Op(name, "add", (a, b), i + 1, length=lengths[a]))
            else:
                src = rng.choice(known)
                start = rng.randrange(0, lengths[src])
                stop = rng.randrange(start + 1, lengths[src] + 1)
                ops.append(
                    Op(name, "slice", (src,), i + 1, start=start, stop=stop,
                       length=stop - start)
                )
            lengths[name] = ops[-1].length
            known.append(name)
        outputs = tuple(rng.sample(known, rng.randint(1, min(2, len(known)))))
        plan = Plan(100000, tuple(inputs), tuple(ops), outputs)

        producer = {s.name: -1 for s in inputs}
        for i, op in enumerate(ops):
            producer[op.name] = i
        valid = []
        for perm in itertools.permutations(range(n_ops)):
            done = set()
            ok = True
            for i in perm:
                for operand in ops[i].operands:
                    if producer[operand] >= 0 and producer[operand] not in done:
                        ok = False
                if not ok:
                    break
                done.add(i)
            if ok:
                valid.append(perm)
        schedule = build_schedule(plan)
        brute = min(peak_for_order(plan, perm) for perm in valid)
        assert schedule.min_elements == brute
        assert schedule.order in valid
        assert peak_for_order(plan, schedule.order) == brute


def test_twelve_op_limit_runs_quickly():
    inputs = "x: 1, 2, 3\n"
    chains = "\n".join(
        [f"a{i} = relu({'x' if i == 0 else f'a{i-1}'})" for i in range(6)]
        + [f"b{i} = relu({'x' if i == 0 else f'b{i-1}'})" for i in range(6)]
    )
    plan = parse_plan(
        {"capacity": "1000", "inputs": inputs, "ops": chains, "outputs": "a5, b5"}
    )
    assert len(plan.ops) == MAX_SCHEDULE_OPS
    start = time.perf_counter()
    schedule = build_schedule(plan)
    assert time.perf_counter() - start < 1.0
    tight = Plan(schedule.min_elements, plan.inputs, plan.ops, plan.outputs)
    assert execute_plan(tight, schedule.order)["ok"] is True


def test_describe_order_carries_positions_and_labels():
    plan = make_plan(
        "x = relu(A)\ny = add(B, C)",
        "A: 1, 2\nB: 3, 4\nC: 5, 6",
        "x, y",
    )
    schedule = build_schedule(plan)
    entries = describe_order(plan, schedule.order)
    assert [e["position"] for e in entries] == [1, 2]
    assert all(e["original_position"] == e["index"] + 1 for e in entries)
    assert any("relu" in e["label"] for e in entries)
    assert any("add(B, C)" in e["label"] for e in entries)


def test_input_loading_order_is_never_reordered():
    # An unused input is released immediately after loading, exactly like the
    # executor does, so it never inflates the op-phase peak.
    plan = make_plan(
        "r = relu(a)",
        "a: 1, 2, 3\nz: 9, 9",
        "r",
    )
    schedule = build_schedule(plan)
    assert schedule.input_peak == 5  # a(3) then z(2) makes 5 before z is freed
    result = execute_plan(plan, schedule.order)
    # z is released immediately after its own loading step (the second step).
    assert result["steps"][1]["freed"][0]["name"] == "z"
