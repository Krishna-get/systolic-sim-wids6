# Week 1 — One PE, Then a Row

**Goal:** by Sunday you will have a working 1×N systolic array that computes a dot product, and you will be able to draw, from memory, where every piece of data sits on any given cycle.

**Time:** ~8 hours. Roughly 2 on reading, 1 on the paper worksheet, 4 on code, 1 on the write-up.

---

## 1. Why any of this exists

Start with the problem the hardware is solving.

A matrix multiply of two N×N matrices does N³ multiply-accumulate operations but only touches 3N² numbers. For N = 1024 that's a billion operations over three million values — **each number gets used about a thousand times.**

A normal CPU doesn't exploit that. It loads operands from memory into registers, computes, writes back. Even with caches, data moves a lot. And moving data is expensive in a way that is easy to underestimate: in a typical process node, reading a 32-bit word from DRAM costs on the order of a hundred times more energy than the multiply you're going to do with it. Not 10% more. Two orders of magnitude.

So the design question for an AI accelerator isn't "how do I do more multiplies per second." It's **"how few times can I move each number?"**

A systolic array's answer: build a grid of tiny compute units, and have them hand data directly to their neighbours. A value is read from memory once, enters the edge of the array, and then gets reused by every cell it passes through without ever going back to memory. The data flows through the array like a pulse — hence *systolic*.

Google's TPU v1 took this to 65,536 PEs in a single 256×256 array. That's the thing you're building a model of.

---

## 2. The processing element

This is the whole of the hardware, and it's almost insultingly simple. One PE holds a weight. Every cycle it:

1. takes an **activation** from its left neighbour
2. takes a **partial sum** from its neighbour above
3. computes `new_sum = partial_sum_in + (weight × activation_in)`
4. passes the activation **right**, unchanged
5. passes `new_sum` **down**

That's it. One multiply, one add, two hand-offs. The weight does not move — this is **weight-stationary** dataflow, the one the TPU uses and the one you'll implement first.

```
            psum_in ↓
               ┌─────────┐
   act_in  →   │  w (held) │   → act_out  (= act_in, delayed 1 cycle)
               └─────────┘
            psum_out ↓   (= psum_in + w × act_in)
```

The crucial, easy-to-miss detail: **both outputs are registered.** `act_out` and `psum_out` do not appear on the same cycle the inputs arrived — they appear on the *next* cycle. That one-cycle delay per hop is what creates the wavefront, and it is the single most common thing people get wrong in Week 1.

### Why registered outputs matter

If data propagated through the whole array combinationally in one cycle, the clock would have to be slow enough for a signal to cross all 256 cells. Registering between every PE means the clock only has to accommodate *one* multiply-add. That's why real systolic arrays clock fast. You are modelling a physical constraint, not an arbitrary software choice.

---

## 3. Cycle-level vs functional simulation

Two things you could write, and they are not the same:

**Functional simulation** answers *what is the result?* That's `np.matmul(A, B)`. One line. Useless for our purposes.

**Cycle-level simulation** answers *what is every register holding at cycle t, and how many cycles does the whole thing take?* That's a loop over time where each iteration updates every PE's state based on the previous cycle's state.

We want the second. The answer is the easy part — the *timing* is the thing we're studying.

### The golden rule of cycle-level simulation

> **Read all of the old state, then write all of the new state. Never mix the two.**

If you update PE 0 and then compute PE 1 using PE 0's *already-updated* value, you've accidentally let data teleport two cells in one cycle. Your dot product will still come out right, which is exactly why this bug is so dangerous — it silently destroys your timing model while leaving the result correct.

Two ways to avoid it, pick one and be consistent:

```python
# Option A: double-buffer the whole array
next_state = [pe.compute(inputs) for pe in self.pes]   # read old
for pe, s in zip(self.pes, next_state):
    pe.commit(s)                                        # then write new

# Option B: iterate in reverse data-flow order
for i in reversed(range(len(self.pes))):                # downstream first
    self.pes[i].tick()
```

Option A is harder to get wrong and generalises to 2D in Week 2. I recommend it.

---

## 4. THE WORKSHEET — do this before writing any code

Open [`worksheet.md`](worksheet.md). **This is graded and it is not optional.**

You will fill in a table, by hand, for a 1×4 array computing a dot product. Columns: cycle number; what's entering from the left; what each of the four PEs holds as `act_in` and `psum_out`; what's leaving the bottom.

It takes about 45 minutes and it is the highest-value 45 minutes of the entire four weeks. Here is why: when you write the code first, you will build something that produces the correct dot product. It will produce the correct dot product even if your timing model is wrong, because addition is commutative and you'll have summed the right numbers in the wrong order at the wrong times. You will then spend Week 2 confused about why your cycle counts don't match the reference, with no intuition to debug against.

Fill in the table first. Then write code that reproduces your table. When they disagree, one of you is wrong, and finding out which is the learning.

**What you should notice while filling it in:**
- The first partial sum doesn't emerge until cycle N (the fill)
- Inputs must be **staggered** — PE 2's activation has to arrive a cycle later than PE 1's, or the sums won't line up
- Near the start and end, some PEs are doing nothing useful

That third observation is utilization, and it is most of Week 2.

---

## 5. The code

### `pe.py`

```python
class ProcessingElement:
    """A single weight-stationary MAC unit."""

    def __init__(self, weight=0):
        self.weight = weight
        self.act_in = 0          # latched input this cycle
        self.psum_in = 0
        self.act_out = 0         # registered output, visible next cycle
        self.psum_out = 0

    def load_weight(self, w):
        """Weight-stationary: set once, held for the whole GEMM tile."""
        # TODO

    def compute(self, act_in, psum_in):
        """Pure function: return (act_out, psum_out) for these inputs.
        Must NOT mutate self — that's what commit() is for."""
        # TODO

    def commit(self, act_out, psum_out):
        """Latch the new outputs. Called after all PEs have computed."""
        # TODO
```

Keeping `compute` pure and `commit` separate is what makes the golden rule above enforceable. Follow this structure even though a single combined `tick()` looks shorter.

### `array_1d.py`

```python
class SystolicRow:
    """N processing elements chained together."""

    def __init__(self, n_pes):
        self.pes = [ProcessingElement() for _ in range(n_pes)]
        self.cycle = 0
        self.trace = []

    def load_weights(self, weights):      # TODO
    def step(self, act_inputs):
        """One cycle. act_inputs[i] is the activation presented to PE i.
        Returns the psum leaving the last PE at the end of this cycle."""
        # TODO
    def run(self, activation_vectors):
        """Run one or more dot products. Returns (results, total_cycles).
        Vector m is injected starting at cycle m — they pipeline."""
        # TODO
    def print_trace(self):
        """Pretty-print the cycle-by-cycle table. Your debugging lifeline."""
        # TODO
```

### How the 1D chain is wired

Activations are injected **directly into each PE** — `x[i]` goes to PE `i`
— rather than rippling in from one end. Partial sums are what chains:
PE `i` takes PE `i-1`'s `psum_out` from the previous cycle, and the
finished dot product emerges from the last PE.

This is **exactly one column of the 2D array you build in Week 2**, where
the activations arrive from the neighbouring column instead of from
outside. Building it this way now means Week 2 is an extension rather than
a rewrite.

Because each hop costs a cycle, `x[i]` must be injected at cycle `i`, not
all at once at cycle 0. That's the stagger, and Exercise 1 of the worksheet
makes you derive it.

Write `print_trace` early and well. You will stare at its output more than any other thing this month. Make it line up in columns.

---

## 6. Checkpoint — due Sunday

Push to `submissions/<your-roll-number>/week1/`:

1. **`worksheet.md`** — your completed hand-trace table.
2. **`pe.py` and `array_1d.py`** — passing all 29 tests:
   ```bash
   pytest week1/test_week1.py -v --impl submissions/<your-roll>/week1
   ```
3. **A trace dump** — output of `print_trace()` for a 1×4 array with weights `[1, 2, 3, 4]` and activations `[5, 6, 7, 8]`.
4. **`answers.md`** — three short written answers:
   - **(a)** Your 1×N array takes more than N cycles to produce one dot product. Where do the extra cycles go, and what are they called?
   - **(b)** If you fed a *second* dot product in immediately after the first (same weights, new activations), how many additional cycles would it take? Why is that number not N again?
   - **(c)** The weight never moves. Name one thing that makes easy and one thing it makes hard.

Answer (b) is the one that separates people who understood pipelining from people who didn't. Think about it before you compute it.

---

## 7. Common mistakes

| Symptom | Almost certainly |
|---|---|
| `test_single_cycle_does_not_propagate` fails with `[5,5,5,5]` | You updated PEs in forward order and let data skip cells. Re-read §3. |
| Result wrong, off by a shifted window | You didn't stagger the inputs. Check your worksheet. |
| Last few outputs are zero / missing | You stopped stepping when inputs ran out. The array still has data in it — keep stepping until the wavefront reaches the end. |
| M vectors cost N×M cycles | You're draining between vectors instead of pipelining. It should be N+M−1. |
| `act_out` equals `act_in` in the same cycle | You forgot the output register. There is a one-cycle delay per hop. |
| Trace is unreadable | Fix it now, not in Week 3. |

---

## 8. Reading

**Do read (≈90 min):**
- **Jouppi et al. (2017), TPU v1 paper** — §3 (architecture) and §4 (the matrix unit). Skip the benchmarking sections for now. Focus on Figure 4 and the description of how weights are loaded.
- **Owumi, "Under the Hood of MatMuls: A Rust Simulation of the TPU's Systolic Array"** (Medium) — someone doing exactly this exercise in another language. Useful sanity check on the PE behaviour, and short.

**Skim, bookmark for Week 3:**
- **Samajdar et al., "SCALE-Sim" (arXiv:1811.02883)** — read §I and §II only. You'll return to §III.B on dataflows.

**If you want the memory-cost claim in §1 properly sourced:**
- **Horowitz, "Computing's Energy Problem" (ISSCC 2014)** — the energy-per-operation table everyone in this field cites.

---

## 9. If you finish early

Don't start Week 2's folder. Instead:

- Make `print_trace` render the array as ASCII art with data positions shown per cycle. Genuinely useful later, and more instructive than it sounds.
- Add an assertion that fires if any PE's `compute()` is called twice in one cycle. You'll be glad of it in Week 2.
- Work out, on paper, how many cycles a 1×N array needs for M consecutive dot products. You'll need the formula next week.
