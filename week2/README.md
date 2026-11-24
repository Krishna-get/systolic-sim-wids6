# Week 2 — The 2D Array and the Mapping Problem

**Goal:** a full R×C systolic array that correctly multiplies matrices of *any* size, plus honest numbers for how much of the hardware you actually used.

**Time:** ~10 hours. This is the heaviest week. 2 on concepts, 5 on code, 2 on experiments, 1 on the write-up.

---

## 1. From a row to a grid

Last week your row computed one dot product: a 1×N vector against an N×1 vector. A matrix multiply is many dot products at once, so we go 2D.

In a weight-stationary R×C array computing `C_out = A @ B` where `A` is M×K and `B` is K×N:

- the array holds a **K×N tile of B** — one weight per PE, so a 32×32 array holds a 32×32 tile of the weight matrix
- rows of **A** stream in from the **left**, one row per array-row, staggered
- **partial sums** flow **downward** through each column and emerge at the bottom
- each column of the array produces one column of the output

Each PE still does exactly what it did last week. Nothing about the PE changes. What changes is that you now have R×C of them and you have to feed them correctly.

```
                  B tile loaded into PEs (stationary)
                 ┌────┬────┬────┬────┐
  A row 0  ───►  │ w  │ w  │ w  │ w  │
                 ├────┼────┼────┼────┤
  A row 1  ───►  │ w  │ w  │ w  │ w  │
                 ├────┼────┼────┼────┤
  A row 2  ───►  │ w  │ w  │ w  │ w  │
                 ├────┼────┼────┼────┤
  A row 3  ───►  │ w  │ w  │ w  │ w  │
                 └─┬──┴─┬──┴─┬──┴─┬──┘
                   ▼    ▼    ▼    ▼
                   output columns
```

### The staggering generalises

In 1D you staggered inputs by one cycle per PE. In 2D you stagger **both** directions: row `i` of A must start entering `i` cycles after row 0, because the partial sums travelling down column `j` need to meet the right activation at the right cycle. Draw this for a 3×3 array before you code it — same discipline as last week's worksheet, and the same payoff.

A clean way to think about it: a value entering at the top-left and a value entering at the bottom-left must *collide* at the correct PE. The array is a space-time schedule, not just a grid.

---

## 2. Fill and drain

Your 1D array had this problem and now it's worse.

For an R×C array, a partial sum entering at the top-left takes R+C−1 cycles to reach the bottom-right. During the first R+C−1 cycles the array is **filling** — some PEs have no valid data yet. During the last R+C−1 cycles it's **draining** — inputs have stopped but results are still working their way out.

Total cycles for one full tile pass, roughly:

```
T ≈ (number of input rows) + R + C − 2
```

For a large matmul streaming many rows, fill/drain is amortised and irrelevant. For a small one it dominates. A 256×256 TPU array needs 511 cycles just to fill — if your layer only has 64 rows to stream, you spend 89% of your time on overhead.

**This is the first real design insight of the project:** big arrays are only efficient on big problems. Hold on to it; Week 4 makes it quantitative.

---

## 3. The mapping problem

Here's the part that makes this a real project rather than an exercise.

Your array is 32×32, fixed in silicon. Your layer's weight matrix is 512×2048. These do not fit. So you **tile**: break the big GEMM into 32×32 chunks, run them one after another, and accumulate.

For `A (M×K) @ B (K×N)` on an R×C array:

```
for k_tile in ceil(K / R):          # reduction dimension
    for n_tile in ceil(N / C):      # output columns
        load B[k_tile, n_tile] into the array    ← costs cycles!
        stream all M rows of A[:, k_tile] through
        accumulate partial results into output[:, n_tile]
```

Three things fall out of this loop, and each one is a lesson:

**(a) Weight-load cost.** Every time you move to a new tile, the array has to be loaded with new weights, which takes R cycles during which no useful work happens. With 2048 tiles, that's a lot of dead time. Real designs hide it by double-buffering the weight registers — loading the next tile's weights while the current tile computes. Model the naive version first, then add double-buffering and measure what you saved.

**(b) Accumulation across k-tiles.** The reduction dimension K is split across tiles, so each tile produces a *partial* result that must be added to the previous tile's. Where does that accumulator live? In an output SRAM buffer. That's traffic. Count it.

**(c) Edge effects.** `ceil()` is doing something expensive. If K = 100 and R = 32, you need 4 tiles, and the last one only uses 4 of 32 rows — 87.5% of the array idles for that entire tile. This is why layer dimensions that are multiples of the array size perform so much better, and it's a large part of why ML engineers are told to pad channel counts to powers of two.

---

## 4. Utilization — define it before you measure it

"Utilization" is used loosely in this field. Pick a definition, state it, use it consistently:

```
utilization = (useful MACs performed) / (PEs × total cycles)
```

Where *useful* excludes: PEs idle during fill/drain, PEs idle during weight load, PEs in the unused portion of a partial tile, and PEs multiplying by padding zeros.

Expect to be shocked. Naive mappings of real layers often land at 20–40%. That number is the whole justification for the field of accelerator mapping research.

In your write-up, break the loss down by cause. "We hit 31% utilization" is a result. "We hit 31%, of which 22 points were lost to partial tiles, 14 to weight-load stalls and 33 to fill/drain" is an analysis, and it's the kind of thing that makes a graph worth looking at.

---

## 5. Convolution: im2col

Week 3 uses CNN layers, so you need the bridge from convolution to matmul.

A conv layer with input `H×W×C_in`, kernel `K×K×C_in×C_out` is not obviously a GEMM. **im2col** makes it one: for each output position, flatten the `K×K×C_in` patch it reads into a row. Stack those rows into a matrix of shape `(H_out · W_out) × (K·K·C_in)`. Flatten the filters into `(K·K·C_in) × C_out`. Now it's a matmul, and your array can run it.

The cost: im2col duplicates input data, by a factor of roughly K² for stride 1. A 3×3 conv blows the input up ~9×. Real accelerators avoid materialising this in memory and do the address generation on the fly — but for a simulator, materialising it is fine and much simpler. Note the memory blow-up in your write-up; it's a real cost you're choosing not to model.

Write `im2col(input_shape, kernel_shape, stride, padding)` that returns just the **resulting GEMM dimensions** (M, K, N). You don't need actual data — you're counting cycles, not computing feature maps. This keeps Week 3 fast.

---

## 6. The code

Extend Week 1. Don't start over.

```python
class SystolicArray:
    def __init__(self, rows, cols, dataflow='ws'):
        self.pes = [[ProcessingElement() for _ in range(cols)]
                    for _ in range(rows)]
        self.rows, self.cols = rows, cols
        self.cycle = 0
        self.stats = Stats()        # your counters live here

    def load_tile(self, weight_tile):
        """Load weights. Costs cycles — count them."""
        # TODO

    def step(self, act_column):
        """One cycle. Golden rule: compute all, then commit all."""
        # TODO

    def run_tile(self, A_tile, B_tile):
        """One tile pass including fill and drain."""
        # TODO

    def run_gemm(self, M, K, N, A=None, B=None):
        """Full tiled GEMM. Returns (result_or_None, stats)."""
        # TODO
```

```python
@dataclass
class Stats:
    total_cycles: int = 0
    compute_cycles: int = 0
    weight_load_cycles: int = 0
    stall_cycles: int = 0
    useful_macs: int = 0
    wasted_macs: int = 0
    sram_reads: int = 0        # start counting now, Week 3 needs it
    sram_writes: int = 0

    @property
    def utilization(self): ...
```

Build `Stats` properly this week. In Week 3 you'll be comparing three dataflows and in Week 4 sweeping dozens of configurations; if the counters are bolted on late they'll be inconsistent and you'll redo the experiments.

### Correctness first

`run_gemm` with `A` and `B` supplied must return a result matching `np.matmul(A, B)` exactly (integers — use integer inputs so there's no floating-point ambiguity). Then, separately, cycle counts must match `reference/expected_outputs/week2_golden.json`.

Both gates. A simulator that gets the right answer with wrong timing is the failure mode this project is built to avoid.

---

## 7. Experiments to run

Produce a table. Array fixed at 32×32 unless stated.

| Experiment | What to vary | What you should see |
|---|---|---|
| Square GEMMs | M=K=N ∈ {32, 64, 128, 256, 512, 1024} | Utilization climbing toward an asymptote — fill/drain amortising |
| Edge effects | K ∈ {32, 33, 64, 65, 96, 100, 128} | Sawtooth. The 33 case is brutal. |
| Skinny GEMMs | M=1024, N=1024, K ∈ {8, 16, 32} | Why batch size 1 inference is painful |
| Array size | 8×8, 16×16, 32×32, 64×64 on a fixed 256³ GEMM | Larger isn't monotonically better |
| Weight-load | naive vs double-buffered, on many small tiles | How much the optimisation is worth |

Plot at least the first two. A sawtooth plot of utilization vs K is the single most convincing artefact you'll produce this week.

---

## 8. Checkpoint — due Sunday

Push to `submissions/<roll>/week2/`:

1. **Working `SystolicArray`** passing `pytest week2/ -v`, including the golden cycle-count tests.
2. **`im2col.py`** returning correct GEMM dimensions for the shapes in `week2/test_im2col.py`.
3. **`results.md`** — the experiment table above, with at least two plots.
4. **`answers.md`**:
   - **(a)** For a 32×32 array and K=100, compute by hand the number of wasted PE-cycles from partial tiles. Does your simulator agree?
   - **(b)** You have a fixed budget of 1024 PEs. Argue for one of: 32×32, 16×64, or 64×16 — for a layer with M=4096, K=64, N=1024. Use your simulator.
   - **(c)** Double-buffering weights removes the load stall. What does it cost in hardware, and when is that cost not worth paying?

Question (b) has no single right answer and is marked on the quality of the argument, not the choice.

---

## 9. Common mistakes

| Symptom | Almost certainly |
|---|---|
| Result correct, cycles ~R+C too low | Forgot drain. The array isn't empty when inputs stop. |
| Result wrong only for non-square matrices | Staggering logic assumes R == C. |
| Utilization > 100% | Counting the same MAC in both `useful` and total, or not counting idle PEs. |
| Utilization suspiciously constant | You're computing it per tile instead of per run. |
| Cycles scale as K×N but not M | You're reloading weights inside the M loop. Weight-stationary means they stay. |
| Everything correct, impossibly slow | Fine. Don't optimise. If a sweep takes 10 min, run it overnight. |

---

## 10. Reading

**Do read:**
- **SCALE-Sim paper (arXiv:1811.02883), §III.A and §III.C** — how they model the array and compute utilization. Compare their definition with yours; if they differ, say so in your write-up.
- **Jouppi et al. TPU paper, §6** — the roofline analysis. Their discussion of why many layers are memory-bound rather than compute-bound is exactly what your utilization numbers are showing.

**Reference as needed:**
- **CS231n notes on convolution** (Stanford) — the clearest short explanation of im2col if §5 moved too fast.
- **"Anatomy of High-Performance Matrix Multiplication"** (Goto & van de Geijn, 2008) — the CPU side of the same tiling problem. Optional, but if you've wondered why BLAS is structured the way it is, this is the paper.

---

## 11. If you finish early

- Implement output double-buffering too, and see whether it matters (prediction: much less than weight double-buffering — check whether you're right).
- Add a `--trace-tile N` flag that dumps the full cycle trace for just one tile of a big GEMM. You'll want it in Week 3.
- Try a non-rectangular mapping: what if you split a K=100 reduction as 32+32+32+4 versus 25+25+25+25? Is the balanced split better? (It isn't, necessarily — work out why.)
