"""Week 1 tests.

Run against the skeleton (will fail until you implement it):
    pytest week1/test_week1.py -v

Run against your own copy:
    pytest week1/test_week1.py -v --impl submissions/<your-roll>/week1

Read the test names when something fails — they tell you which concept
is wrong, not just which line.
"""

import random

import pytest

from pe import ProcessingElement
from array_1d import SystolicRow


# ----------------------------------------------------------------------
# The processing element
# ----------------------------------------------------------------------

def test_pe_loads_weight():
    pe = ProcessingElement()
    pe.load_weight(7)
    assert pe.weight == 7


def test_pe_computes_mac():
    pe = ProcessingElement()
    pe.load_weight(3)
    act_out, psum_out = pe.compute(act_in=4, psum_in=10)
    assert psum_out == 22          # 10 + 3*4
    assert act_out == 4            # activation passes through unchanged


def test_pe_compute_is_pure():
    """compute() must not mutate the PE.

    If this fails you have merged compute and commit, and the array will
    let data skip cells. See week1/README.md §3.
    """
    pe = ProcessingElement()
    pe.load_weight(3)
    before = (pe.act_out, pe.psum_out)
    pe.compute(act_in=4, psum_in=10)
    assert (pe.act_out, pe.psum_out) == before, \
        "compute() changed the PE's registers; it must be a pure function"


def test_pe_outputs_only_visible_after_commit():
    pe = ProcessingElement()
    pe.load_weight(3)
    out = pe.compute(act_in=4, psum_in=10)
    assert pe.psum_out == 0, "output appeared before commit — no register"
    pe.commit(*out)
    assert pe.psum_out == 22


def test_pe_weight_is_stationary():
    """The weight survives many compute/commit cycles."""
    pe = ProcessingElement()
    pe.load_weight(5)
    for _ in range(10):
        pe.commit(*pe.compute(1, 0))
    assert pe.weight == 5


# ----------------------------------------------------------------------
# The array — structure
# ----------------------------------------------------------------------

def test_array_has_n_pes():
    row = SystolicRow(8)
    assert len(row.pes) == 8


def test_load_weights_rejects_wrong_length():
    row = SystolicRow(4)
    with pytest.raises(Exception):
        row.load_weights([1, 2, 3])


def test_load_weights_assigns_in_order():
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    assert [pe.weight for pe in row.pes] == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# The array — cycle-by-cycle timing
#
# This is the important block. These tests check the INTERNAL STATE after
# every cycle, not just the final answer, because a wrong implementation
# usually still produces the right answer.
# ----------------------------------------------------------------------

def test_single_cycle_does_not_propagate():
    """After one cycle only PE 0 should hold a partial sum.

    If you get [5, 5, 5, 5] you are computing and committing PE by PE in
    forward order, so PE 0's brand-new output reached PE 3 in the same
    cycle. Data can only move one PE per cycle.
    """
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    row.step([5, 0, 0, 0])
    assert [pe.psum_out for pe in row.pes] == [5, 0, 0, 0]


def test_wavefront_advances_one_pe_per_cycle():
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])

    expected = [
        [5, 0, 0, 0],      # after cycle 0
        [0, 17, 0, 0],     # after cycle 1:  5 + 2*6
        [0, 0, 38, 0],     # after cycle 2: 17 + 3*7
        [0, 0, 0, 70],     # after cycle 3: 38 + 4*8
    ]
    acts = [
        [5, 0, 0, 0],
        [0, 6, 0, 0],
        [0, 0, 7, 0],
        [0, 0, 0, 8],
    ]
    for c, (a, e) in enumerate(zip(acts, expected)):
        row.step(a)
        got = [pe.psum_out for pe in row.pes]
        assert got == e, f"cycle {c}: expected {e}, got {got}"


def test_step_returns_value_leaving_last_pe():
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    outs = [row.step(a) for a in ([5, 0, 0, 0], [0, 6, 0, 0],
                                  [0, 0, 7, 0], [0, 0, 0, 8])]
    assert outs == [0, 0, 0, 70]


# ----------------------------------------------------------------------
# The array — dot products
# ----------------------------------------------------------------------

def test_single_dot_product():
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    result, cycles = row.run_single([5, 6, 7, 8])
    assert result == 70
    assert cycles == 4, "an N-PE array needs N cycles for one dot product"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 32])
def test_dot_product_various_sizes(n):
    rng = random.Random(0xC0FFEE + n)
    w = [rng.randint(-9, 9) for _ in range(n)]
    x = [rng.randint(-9, 9) for _ in range(n)]

    row = SystolicRow(n)
    row.load_weights(w)
    result, cycles = row.run_single(x)

    assert result == sum(wi * xi for wi, xi in zip(w, x))
    assert cycles == n


def test_pipelined_dot_products():
    """Consecutive dot products cost ONE extra cycle each, not N.

    This is the whole point of a systolic array. If you get N*M cycles,
    you are draining the array between vectors instead of pipelining.
    """
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    vectors = [[5, 6, 7, 8], [1, 1, 1, 1], [2, 0, 2, 0]]
    results, cycles = row.run(vectors)

    assert results == [70, 10, 8]
    assert cycles == 4 + 3 - 1


@pytest.mark.parametrize("n,m", [(4, 1), (4, 5), (8, 3), (16, 16), (3, 7)])
def test_pipelined_various(n, m):
    rng = random.Random(n * 1000 + m)
    w = [rng.randint(-9, 9) for _ in range(n)]
    vectors = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(m)]

    row = SystolicRow(n)
    row.load_weights(w)
    results, cycles = row.run(vectors)

    expected = [sum(wi * xi for wi, xi in zip(w, v)) for v in vectors]
    assert results == expected
    assert cycles == n + m - 1


def test_zero_weights_give_zero():
    row = SystolicRow(4)
    row.load_weights([0, 0, 0, 0])
    result, _ = row.run_single([5, 6, 7, 8])
    assert result == 0


def test_reset_clears_state_but_keeps_weights():
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    row.run_single([5, 6, 7, 8])
    row.reset()
    assert all(pe.psum_out == 0 for pe in row.pes)
    assert [pe.weight for pe in row.pes] == [1, 2, 3, 4]
    # and it still works afterwards
    result, _ = row.run_single([5, 6, 7, 8])
    assert result == 70


# ----------------------------------------------------------------------
# Tracing
# ----------------------------------------------------------------------

def test_trace_records_every_cycle():
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    row.run_single([5, 6, 7, 8])
    assert len(row.trace) == 4


def test_print_trace_runs(capsys):
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    row.run_single([5, 6, 7, 8])
    row.print_trace()
    out = capsys.readouterr().out
    assert "70" in out, "your trace should show the final result"
    assert len(out.strip().splitlines()) >= 4
