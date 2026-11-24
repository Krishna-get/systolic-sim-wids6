# Inside the TPU: Building a Systolic Array Simulator from Scratch

**WiDS 6.0 | Analytics Club, IIT Bombay**
**Mentor:** Abhiraj Patel
**Duration:** 4 weeks | **Commitment:** 8–10 hrs/week | **Format:** Guided Project

---

## What this project is

Every matrix multiply in a neural network eventually runs on hardware. On Google's TPUs, that hardware is a **systolic array** — a 2D grid of tiny processing elements that pass data to each other in lockstep, one clock cycle at a time, like a heartbeat. ("Systolic" comes from *systole*, the contraction of the heart.)

Over four weeks you will build a cycle-level simulator of one, in pure Python, starting from a single processing element and ending with a design-space exploration tool whose numbers you validate against SCALE-Sim, the simulator used in published accelerator research.

You will not train a model. You will not touch a GPU. You will find out **why** the models you already train are shaped the way they are.

---

## What you'll be able to do at the end

- Explain what a systolic array is and trace data through one by hand
- Map an arbitrary matrix multiply onto a fixed-size PE array, including the awkward edges
- Implement and compare weight-stationary, input-stationary and output-stationary dataflows
- Quantify utilization, SRAM traffic and DRAM stalls for real network layers
- Make an argument about accelerator design backed by numbers you generated yourself
- Read an accelerator architecture paper and know what the graphs mean

---

## The four weeks

| Week | Theme | You build | You learn |
|------|-------|-----------|-----------|
| 1 | One PE, then a row | A single MAC unit, chained into a 1×N array | What "systolic" actually means; the wavefront |
| 2 | The 2D array and the mapping problem | Full R×C array, tiled GEMM for any matrix size | Fill/drain latency, tiling, utilization, edge waste |
| 3 | Dataflows | WS, IS and OS mappings + SRAM traffic counters | Why reuse strategy is the central design choice |
| 4 | Design space exploration | Array/SRAM sweeps, DRAM stall model, SCALE-Sim validation | That compute-only models lie |

Each week has its own folder with a full README, starter code, tests and a reading list. **Start at [`week1/README.md`](week1/README.md).**

---

## Prerequisites

**Required**
- Python: functions, classes, nested loops, list/dict comprehension. You will write a lot of plain Python.
- NumPy: array creation, indexing, slicing, `@` / `np.matmul`.
- Linear algebra: you know what a matrix multiply does, index by index.

**Helpful but not required**
- Any NN framework (PyTorch/TF) — only for reading layer shapes in Week 3.
- matplotlib — for Week 4 plots. You can learn it that week.

**Explicitly NOT required**
- Verilog, CUDA, computer architecture coursework, any hardware background at all. Everything about accelerators is taught here from first principles. If you have done a digital-design course you'll move faster in Week 1; you will not be ahead by Week 3.

---

## Setup

```bash
git clone <repo-url>
cd systolic-sim-wids6
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest week1/                   # should collect tests and fail — that's correct on day 1
```

`requirements.txt` is deliberately tiny: `numpy`, `pytest`, `matplotlib`, and from Week 4, `scalesim`.

Runs on any laptop. **No GPU, no Colab, no cloud account needed.** This is deliberate — the entire project is CPU-bound Python and the largest simulation in Week 4 takes a couple of minutes.

---

## Repo structure

```
systolic-sim-wids6/
├── README.md               ← you are here
├── requirements.txt
├── week1/
│   ├── README.md           instructions, concepts, reading
│   ├── worksheet.md        hand-trace exercise (graded)
│   ├── pe.py               skeleton with TODOs
│   ├── array_1d.py         skeleton with TODOs
│   └── test_week1.py       tests you must pass
├── week2/  …
├── week3/  …
├── week4/  …
├── reference/
│   ├── expected_outputs/   golden cycle counts for grading
│   └── topologies/         layer-shape CSVs (ResNet-18, BERT, ViT)
└── submissions/
    └── <your-roll-number>/ your work goes here
```

---

## How submissions work

Each week has a **checkpoint** — specific tests to pass plus a short written answer. Push to your own branch (`wids6/<your-roll-number>`) by the Sunday deadline and open a PR against `main`. I review every PR individually; that's the main feedback channel.

The written answers matter as much as the code. A simulator that produces the right number but whose author can't explain *why* that number is right has not finished the exercise.

**Correctness gate:** from Week 2 onward, your simulator must reproduce the golden cycle counts in `reference/expected_outputs/` before any analysis you do counts. A subtly wrong simulator produces plausible-looking graphs, which is the single biggest risk in this kind of project.

---

## Communication

- **Weekly call:** 1 hour, recorded. Concept walkthrough + live debugging of whatever people are stuck on.
- **Group channel:** for anything. Please ask publicly rather than DMing — four other people are stuck on the same thing.
- **Office hours:** by appointment, for anyone genuinely blocked for more than a day. Being stuck for a day is normal and productive. Being stuck for three is not; come talk to me.

---

## A note on how to work on this

Resist the urge to make it fast. This simulator models hardware one cycle at a time, and a readable `for cycle in range(total_cycles):` loop that you can print traces out of is worth more to you than a clever vectorised version you can't debug. Optimisation is not the point; **understanding what the hardware is doing on cycle 47 is the point.**

The other thing: do the paper exercises. Week 1's hand-trace looks like busywork and is the single most important hour of this project. Everyone who skips it builds an array that produces correct answers by accident — a normal matmul with extra steps — without ever seeing the wavefront.

---

## Reading, overall

Three papers anchor the whole project. Read the first one in Week 1; the others arrive when they're needed.

1. **Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017)** — the TPU v1 paper. The original motivation for all of this.
2. **Samajdar et al., "SCALE-Sim: Systolic CNN Accelerator Simulator" (arXiv:1811.02883)** — the simulator you're building a small version of. Sections on dataflow are the ones that matter.
3. **Chen et al., "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs" (JSSC 2017)** — where the dataflow taxonomy comes from.

Everything else is linked from the relevant week.
