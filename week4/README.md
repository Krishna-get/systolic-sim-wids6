# Week 4 — Design Space Exploration, DRAM, and Validation

**Goal:** turn your simulator into a design tool, discover that last week's conclusion was wrong, and prove your numbers are trustworthy by checking them against a published simulator.

**Time:** ~10 hours. 3 on the DRAM model, 3 on sweeps, 2 on validation, 2 on the final report.

---

## 1. The lie in your Week 3 results

Last week you concluded that one dataflow was better. That conclusion was built on an assumption you didn't state: **that data is always available when the array wants it.**

It isn't. Your array reads from on-chip SRAM, and SRAM is small — a few hundred KB, because it's expensive in area and leakage. The actual weights and activations live in DRAM, off-chip, with bandwidth measured in tens of GB/s and latency measured in hundreds of cycles. When SRAM doesn't have what the array needs, the array **stalls**.

And here's the thing: the dataflow that minimises compute cycles is not necessarily the one that minimises DRAM traffic. Often it's the opposite, because reducing compute cycles usually means moving more data.

The SCALE-Sim v3 authors document exactly this inversion. Weight-stationary shows about 21% fewer compute cycles than output-stationary across six ResNet-18 layers — but once DRAM stalls are modelled, output-stationary comes out roughly 30% *ahead* on total execution cycles. Same hardware, same layers, opposite answer.

Your job this week is to reproduce that reversal with your own simulator. **Do not look up the answer and fit to it.** Build the model, run it, and see what you get.

---

## 2. Building the memory model

You need three levels. Keep it simple — a few parameters, not a full DRAM timing model.

### Level 1: SRAM (on-chip buffers)

Three separate buffers, as real designs have:

```python
@dataclass
class SRAMConfig:
    ifmap_kb: int = 128
    filter_kb: int = 128
    ofmap_kb: int = 128
    read_bw_bytes_per_cycle: int = 64
    write_bw_bytes_per_cycle: int = 64
```

The array requests operands each cycle. If the buffer holds them, fine. If not, a **miss** — the data must come from DRAM, and the array stalls until it arrives.

### Level 2: DRAM

```python
@dataclass
class DRAMConfig:
    bandwidth_gbps: float = 25.6      # typical LPDDR4/HBM-lite
    latency_cycles: int = 100
    clock_ghz: float = 1.0
```

Convert bandwidth to bytes/cycle: `bandwidth_gbps / clock_ghz` gives bytes per cycle at your clock. A fetch of `n` bytes takes `latency_cycles + n / bytes_per_cycle` — latency once, then bandwidth-limited streaming.

### Level 3: the stall model

```python
def fetch(self, n_bytes):
    """Returns cycles the array must stall."""
    if self.sram.has(n_bytes):
        return 0
    cycles = self.dram.latency + n_bytes / self.dram.bytes_per_cycle
    self.stats.dram_bytes += n_bytes
    self.stats.stall_cycles += cycles
    return cycles
```

Then: `total_cycles = compute_cycles + stall_cycles`.

**Simplifications you are explicitly allowed to make** (state them in your report):
- No DRAM row-buffer modelling, bank conflicts, or refresh
- No prefetching unless you choose to add it
- Perfect LRU in SRAM
- No contention between the three buffers

These are real inaccuracies. SCALE-Sim v3 added a proper DRAM model (via Ramulator) precisely because the simplified version wasn't good enough for their conclusions. Naming your model's limits is part of doing this properly — a simulator paper that doesn't list its simplifications is not trustworthy.

### Optional but valuable: double-buffering

Add a flag that overlaps the next tile's DRAM fetch with the current tile's compute. This is what real designs do and it dramatically changes the stall picture. Running with it on and off is one of your better experiments.

---

## 3. The design space exploration

Now sweep. Use the same layer set as Week 3.

**Sweep 1 — array size.** 8×8, 16×16, 32×32, 64×64, 128×128, at fixed SRAM. Plot total cycles and utilization.

Expect latency to improve with size and utilization to collapse. The SCALE-Sim v3 authors report a 128×128 array being 6.53× faster than 32×32 on ViT-base measured on latency alone — but 32×32 being 2.86× *more energy-efficient*, thanks to better utilization and lower leakage. If you add the energy model below, see whether you reproduce that shape.

**Sweep 2 — SRAM size.** 32 KB to 1 MB per buffer, array fixed at 32×32. Find the knee: the point past which more SRAM buys nothing because the working set already fits. The knee's location depends on the layer, and that is itself the finding.

**Sweep 3 — DRAM bandwidth.** 8, 16, 25.6, 51.2, 102.4 GB/s. Identify which layers are compute-bound (flat line) and which are memory-bound (slope). This is a roofline analysis in all but name; say so.

**Sweep 4 — the money shot.** All three dataflows, compute-only vs compute+DRAM, side by side. This is where the inversion either shows up or doesn't.

**Sweep 5 — iso-compute.** One 64×64 array versus four 32×32 arrays (same PE count). Which wins, and does the answer depend on the layer? SCALE-Sim v3 found that multi-core arrangements narrow the gap between dataflows considerably — WS beats IS by 1.87× single-core but only 1.14× across 16 smaller cores.

### Optional: a first-order energy model

If you want the energy results, add relative costs (normalised to one MAC):

```python
E_MAC        = 1
E_SRAM_READ  = 6
E_SRAM_WRITE = 6
E_DRAM       = 200
```

These are rough but the right order of magnitude — the point is that DRAM dominates by two orders, which is the only thing the model needs to get right to be useful. Cite Horowitz (ISSCC 2014) for the real numbers.

Energy flips conclusions that latency doesn't. Report both.

---

## 4. Validation against SCALE-Sim

This is what separates a toy from a tool.

SCALE-Sim is the open-source systolic array simulator from Georgia Tech and ARM Research — the one the papers you've been reading use. It's written entirely in Python, installs as a package, and runs from a config file plus a topology CSV.

```bash
pip install scalesim
python -m scalesim.scale -c configs/32x32_ws.cfg -t topologies/resnet18.csv -p outputs/
```

The config file sets array dimensions, SRAM sizes and dataflow; the topology CSV is the same layer format you've been using.

**Your task:** run SCALE-Sim and your simulator on the same configuration and compare, layer by layer.

| Config | Your cycles | SCALE-Sim cycles | % diff |
|---|---|---|---|
| 32×32 WS, ResNet-18 L1 | | | |
| … | | | |

**What counts as success:** within about 10% on compute cycles for most layers. You will not match exactly, and you shouldn't expect to — SCALE-Sim models things you don't. The valuable part is **explaining each discrepancy**:

- Larger gap on layers with awkward dimensions → different edge-tile handling
- Consistent offset of a few hundred cycles → different fill/drain accounting
- Your numbers systematically lower → you're probably not counting weight-load stalls the way they do
- Wildly different on one layer only → you have a bug, find it

A table of discrepancies with a one-line explanation for each is the strongest single artefact in your final report. "My simulator is 7% optimistic because I don't model the weight FIFO" is a much better sentence than "my simulator matches."

---

## 5. Final deliverable

Push to `submissions/<roll>/final/`:

### 1. The simulator
Clean, documented, with a README explaining how to run it. Someone should be able to clone it and reproduce your Week 4 figures with one command. Include a `run_all.py`.

### 2. The report (`report.md` or PDF, 6–10 pages)

Structure it like a short paper:

- **Abstract** — three sentences. What you built, what you measured, what you found.
- **Background** — systolic arrays and dataflows, in your own words, briefly.
- **Simulator design** — architecture, what you model, and an explicit list of what you don't.
- **Validation** — the SCALE-Sim comparison table with explanations.
- **Results** — the sweeps, with plots. One finding per subsection, each with a claim in the heading.
- **Discussion** — the compute-only vs with-DRAM inversion. What would you design, and for what workload?
- **Limitations and future work** — honest.

### 3. A five-minute presentation
For the final session. One slide per finding, maximum. Lead with the inversion result; it's your best material.

---

## 6. What makes a good report here

The simulator being correct is the floor, not the achievement. What distinguishes the good submissions:

**Claims in headings.** "Utilization collapses beyond 64×64 for ResNet's late layers" beats "Results of Sweep 1."

**Every plot answers a question.** If you can't write the question above the figure, cut the figure.

**Numbers you can defend.** For any number in the report, you should be able to say where it came from and roughly reproduce it by hand for a simple case. Expect to be asked.

**The disagreement is interesting.** If your results differ from the published ones, that's a finding, not a failure — provided you investigate. "I get the opposite result and here's why I think my model differs" is excellent work. "I got a different number so I tweaked a constant" is not.

**Limitations stated up front.** The most common weakness in student accelerator reports is over-claiming from a simplified model. You modelled a lot and you simplified a lot; be explicit about which is which.

---

## 7. Common mistakes

| Symptom | Almost certainly |
|---|---|
| No inversion appears | Your DRAM model is too generous, or you're not counting the OS accumulator saving. Check bytes-moved per dataflow first. |
| Stalls dominate everything | Bandwidth units wrong — GB/s vs bytes/cycle conversion. Sanity-check that one cycle can move a sensible number of bytes. |
| SCALE-Sim won't run | The topology CSV is format-sensitive; use their provided examples as the template before your own. |
| Your cycles 2–3× off SCALE-Sim | Dataflow mismatch in the config, or you're comparing their total to your compute-only. |
| All sweeps look flat | SRAM is large enough to hold everything. Shrink it until something interesting happens. |
| Energy results look wrong | You're probably counting SRAM accesses but not DRAM, or vice versa. |

---

## 8. Reading

**Do read:**
- **Raj et al., "SCALE-Sim v3" (arXiv:2504.15377)** — the central paper for this week. Their results tables are exactly the experiments you're running, including the DRAM inversion and the array-size energy trade-off. Read it *after* you've got your own numbers, so you're comparing rather than copying.
- **Williams et al., "Roofline: An Insightful Visual Performance Model" (CACM 2009)** — Sweep 3 is a roofline analysis. Twenty minutes, and it gives you the right vocabulary for the discussion section.

**Reference:**
- **SCALE-Sim documentation and GitHub** — for the config/topology formats.
- **Horowitz, "Computing's Energy Problem" (ISSCC 2014)** — for the energy constants.

---

## 9. Where to go after this

If this project landed well, the natural next steps:

- **Write the array in Verilog.** You now understand the dataflow completely, which is the hard part; the RTL is mechanical by comparison. Berkeley's Gemmini is an open-source systolic accelerator worth reading.
- **Model sparsity.** Skipping zero weights on a systolic array is genuinely hard — the array is rigid and a skipped MAC leaves a hole in the schedule. Why it's hard is a better question than how to do it.
- **Go the GPU route instead.** Write the same GEMM as a tiled CUDA kernel and compare what the hardware gives you for free versus what you had to schedule by hand. This is the direct bridge back to the GPU side of data science.
- **Read a real accelerator paper and know what it means.** Try the TPU v4 paper, or anything from ISCA/MICRO. You now have the vocabulary for the entire results section, which is most of what was blocking you before.
