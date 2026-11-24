# Week 3 — Dataflows

**Goal:** implement all three classic dataflows, run real network layers through each, and produce a comparison that shows there is no universal winner.

**Time:** ~9 hours. 2 on concepts, 4 on code, 2 on experiments, 1 on the write-up.

---

## 1. The central question

Your array has R×C processing elements. A GEMM has three loops — M, K, N. You have two spatial dimensions and three loop dimensions, so **something has to be mapped to time rather than space**, and you get to choose what.

That choice is the **dataflow**, and it determines which matrix gets reused inside the array and which has to keep being fetched from memory. The terminology comes from the Eyeriss paper (Chen et al.) and it's the standard vocabulary in the field. As the SCALE-Sim authors put it, the "stationary" term names the matrix that is pinned to a given PE.

Three choices, three different machines:

| Dataflow | Pinned in each PE | Streams through | Reduction happens |
|---|---|---|---|
| **Weight Stationary (WS)** | one weight | activations ↔, partial sums ↕ | spatially, down columns |
| **Output Stationary (OS)** | one output element (accumulator) | weights and activations both stream in | temporally, in place |
| **Input Stationary (IS)** | one activation | weights stream, partial sums flow | spatially |

Same hardware, same answer, wildly different memory traffic and cycle counts.

---

## 2. Weight Stationary — what you already built

Each PE holds one weight for the duration of a tile. Activations enter from the left and move right; partial sums move down and are reduced as they go, emerging complete at the bottom.

**Reuses:** weights. Each weight is loaded once per tile and used for every row of A that streams past.
**Costs:** reloading the array on every new weight tile (the stall you modelled in Week 2). Partial sums move a lot.
**Good when:** the weight matrix is small relative to the activation volume — large batch sizes, or convolutions where one filter is applied to many positions.
**This is what the TPU v1 does.**

## 3. Output Stationary

Each PE owns one element of the output matrix and holds a running accumulator. Both weights and activations stream in, from left and top. Over K cycles, the PE accumulates the full dot product in place, then the finished output is read out.

**Reuses:** partial sums — and this is the big one, because partial sums are typically wider than inputs. If activations are INT8 but accumulators are INT32, keeping the accumulator still saves 4× the bits per movement. (This is exactly the mixed-precision arrangement real NPUs use, and why INT8×INT8→INT32 is the standard datapath.)
**Costs:** both inputs must be fetched repeatedly. No input reuse inside the array.
**Good when:** the reduction dimension K is long, so the in-place accumulation pays off over many cycles.

## 4. Input Stationary

Each PE holds one activation; weights stream through. The mirror image of WS.

**Reuses:** activations.
**Good when:** the same input feature map is used by many filters — a layer with few input channels and many output channels.
**Least used of the three in practice**, but implementing it is cheap once you have the other two and it makes the comparison complete.

---

## 5. What you're actually counting

The cycle count is the easy half. The interesting half is memory traffic, and you need three counters, maintained for each dataflow:

```python
stats.sram_reads_ifmap     # input feature map / activation reads
stats.sram_reads_filter    # weight reads
stats.sram_writes_ofmap    # output writes
```

The reason the dataflows differ is almost entirely in these numbers. A dataflow that reuses weights reads filters once per tile; one that doesn't reads them once per *row*. The ratio can be two orders of magnitude.

**Also track the bit-width.** Count bytes, not just accesses. An INT32 partial sum write costs 4× an INT8 activation read, and if you only count accesses you will draw the wrong conclusion about output-stationary.

---

## 6. Real layers

Stop testing on square matrices. `reference/topologies/` has CSVs in SCALE-Sim's format — one row per layer, with the dimensions:

```
Layer, IFMAP_H, IFMAP_W, Filter_H, Filter_W, Channels, Num_Filters, Stride
```

Three networks provided:

- **ResNet-18** — convolutions of varying shape. Early layers: large spatial, few channels. Late layers: small spatial, many channels. The two halves behave completely differently and that's the point.
- **BERT-base** — pure GEMMs, large and square-ish. The easy case.
- **ViT-base** — attention plus MLP blocks. Mixed.

Run every layer of each network under all three dataflows. That's 3 × (≈20 + ≈12 + ≈12) ≈ 130 simulations. Script it; don't run them by hand.

Use the same `im2col` from Week 2 to turn conv layers into GEMM dimensions.

---

## 7. What you should find

I'm telling you the expected results so you can tell whether your simulator is broken. Don't take them on trust — *your* numbers are the deliverable, and if they disagree, investigate rather than fudge.

**Expected finding 1 — WS wins on compute cycles.** Across ResNet layers, weight-stationary should come out meaningfully ahead of output-stationary on pure compute cycles. SCALE-Sim's authors report around a 21% reduction in compute cycles for WS over OS across six ResNet-18 layers. If you're in that neighbourhood, your model is behaving.

**Expected finding 2 — it's layer-dependent.** The aggregate hides enormous variation. Early ResNet layers (big spatial, 3–64 channels) and late ones (7×7 spatial, 512 channels) will rank the dataflows differently. Report per-layer, not just totals. A single bar chart of "WS vs OS" is a worse deliverable than a per-layer plot that shows the crossover.

**Expected finding 3 — your conclusion is premature.** You are currently counting compute only. Week 4 adds DRAM, and the ranking changes. Write your Week 3 conclusion confidently anyway; it's more instructive to be publicly wrong and then corrected by your own data.

---

## 8. The code

Refactor rather than duplicating. You want one array class with pluggable mapping, not three copies:

```python
class Dataflow(ABC):
    @abstractmethod
    def schedule(self, M, K, N, rows, cols):
        """Yield the tile sequence and per-tile cycle cost."""

    @abstractmethod
    def traffic(self, M, K, N, rows, cols):
        """Return (ifmap_bytes, filter_bytes, ofmap_bytes)."""

class WeightStationary(Dataflow): ...
class OutputStationary(Dataflow): ...
class InputStationary(Dataflow): ...
```

Then:

```python
array = SystolicArray(32, 32, dataflow=OutputStationary())
stats = array.run_layer(layer_spec)
```

The OS array needs a different PE — one with an internal accumulator register that persists across cycles, rather than passing partial sums down. Subclass `ProcessingElement`; don't add a mode flag with branches everywhere.

### Sanity checks that catch real bugs

- All three dataflows must produce **the same numerical result** for the same inputs. If they don't, one mapping is wrong. This test is worth more than all the others combined.
- Total useful MACs must be identical across dataflows (it's the same arithmetic). Only *cycles* and *traffic* should differ. If `useful_macs` differs, you have a mapping bug.

---

## 9. Checkpoint — due Sunday

Push to `submissions/<roll>/week3/`:

1. **Three working dataflows** passing `pytest week3/ -v`, including the cross-dataflow equivalence test.
2. **`results/` folder** with a CSV of all ~130 runs: network, layer, dataflow, cycles, utilization, bytes by category.
3. **Three plots minimum:**
   - Per-layer cycles, three dataflows, grouped bars, for ResNet-18
   - Total bytes moved by category, stacked, per dataflow
   - Utilization vs layer index, showing how it evolves through the network
4. **`answers.md`**:
   - **(a)** Which dataflow wins on your numbers, and by how much? State the metric you're using.
   - **(b)** Find one layer where the ranking reverses. Explain the reversal in terms of that layer's shape.
   - **(c)** Output-stationary keeps INT32 accumulators in the PEs instead of moving them. Estimate the bytes saved for one ResNet layer. Why doesn't that saving win the overall comparison?
   - **(d)** You're designing an accelerator for *one* workload: BERT inference at batch size 1. Which dataflow, and what array shape? Defend it with your data.

---

## 10. Common mistakes

| Symptom | Almost certainly |
|---|---|
| Dataflows give different results | Mapping bug. Debug on a 4×4 array with a 4×4×4 GEMM and print traces. |
| OS has identical traffic to WS | You didn't actually change where the accumulator lives. |
| Traffic counts are suspiciously round | You're computing them analytically instead of counting in the simulation loop. Count them for real — the analytical formula misses edge tiles. |
| Conv layers all report the same utilization | im2col is returning wrong dimensions; check a layer by hand. |
| One dataflow is 100× slower | Probably correct, actually. IS on a layer with few input channels is genuinely terrible. Verify, then report it. |

---

## 11. Reading

**Do read, properly:**
- **Chen et al., "Eyeriss" (JSSC 2017), §II** — the source of the dataflow taxonomy. They also describe row-stationary, a fourth option that doesn't map cleanly onto a systolic array; read it and you'll understand why the three you implemented are the three SCALE-Sim models.
- **SCALE-Sim paper, §III.B** — their description of OS, WS and IS in systolic-array terms specifically. Figure 2 is the one to study. Compare their scheduling with yours.

**Worth a skim:**
- **Raja, "Systolic Array Data Flows for Efficient Matrix Multiplication in DNNs" (arXiv:2410.22595)** — a short paper doing roughly your Week 3 experiment with SCALE-Sim. Useful as a comparison point for your numbers; note that someone published this, which should tell you your deliverable is real work.

---

## 12. If you finish early

- Add a **mixed-dataflow** mode: pick the best dataflow per layer rather than one for the whole network. How much does the oracle gain over the best fixed choice? (This is an active research question, not a toy exercise.)
- Model **weight sparsity**: if 60% of weights are zero, can any dataflow skip them? Which one makes that easiest, and why is sparsity so hard on a systolic array specifically?
- Start Week 4's reading early — the SCALE-Sim v3 paper is the one that will upend your conclusions.
