# Week 1 Worksheet — Hand-Tracing the Wavefront

**Graded. Do this before you write any code.**

Fill it in by hand — on paper, on a tablet, in this file, doesn't matter.
About 45 minutes. Submit it as `submissions/<your-roll>/week1/worksheet.md`.

---

## Why this is mandatory

You are about to write a program that produces a number. The number will be
correct even if your timing model is completely wrong, because addition is
commutative — you will have summed the right values at the wrong times and
got away with it.

Then in Week 2 your cycle counts won't match the reference, you'll have no
intuition to debug against, and you'll be reverse-engineering your own code.

Fill in the table first. Then write code that reproduces your table. When
they disagree, one of you is wrong, and finding out which one is the entire
point of this exercise.

---

## The setup

A chain of **4 processing elements**.

- PE *i* permanently holds weight `w[i]`. Weights: **w = [1, 2, 3, 4]**
- Activation `x[i]` is injected **directly into PE i** from outside.
  Activations: **x = [5, 6, 7, 8]**
- Partial sums **chain along the array**: PE *i* receives PE *(i−1)*'s
  `psum_out` from the **previous** cycle. PE 0 always receives 0.
- Every PE does: `psum_out = psum_in + w × act_in`
- **Both outputs are registered** — the result of cycle *c* is visible to
  the next PE on cycle *c+1*, never on cycle *c*.

```
   x[0]       x[1]       x[2]       x[3]          ← injected from outside
    ↓          ↓          ↓          ↓
  ┌────┐     ┌────┐     ┌────┐     ┌────┐
0→│ w=1│ ──► │ w=2│ ──► │ w=3│ ──► │ w=4│ ──► out
  └────┘     └────┘     └────┘     └────┘
           partial sums flow left to right
```

The answer should be **1×5 + 2×6 + 3×7 + 4×8 = 70**.

---

## Exercise 1 — the stagger

Here is the trap. Suppose you inject **all four activations on cycle 0**:

| cycle | x into PE0 | x into PE1 | x into PE2 | x into PE3 |
|---|---|---|---|---|
| 0 | 5 | 6 | 7 | 8 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |

**Q1.1** Trace this. What value eventually leaves PE 3, and why isn't it 70?

> *(your answer)*

**Q1.2** The partial sum takes one cycle to move between PEs. So on cycle 1,
the psum from PE 0 arrives at PE 1 — but PE 1's activation was presented on
cycle 0 and is gone. Write down the schedule that fixes this: on which cycle
must `x[i]` be injected into PE *i*?

> *(your answer)*

---

## Exercise 2 — the trace table

Using the correct stagger from Q1.2, fill in every cell.

`act_in` = the activation presented to that PE on that cycle.
`psum_out` = what that PE has **latched at the end of** that cycle.

| cycle | act PE0 | act PE1 | act PE2 | act PE3 | psum PE0 | psum PE1 | psum PE2 | psum PE3 | leaves array |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0 | 0 | 0 |  |  |  |  |  |
| 1 |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |  |

**Q2.1** On which cycle does 70 leave the array?

> *(your answer)*

**Q2.2** Count the PE-cycles. There are 4 PEs × 4 cycles = 16 PE-cycles
available. How many of them did useful work (multiplied a non-zero
activation)? What fraction is that?

> *(your answer)*

**Q2.3** That fraction is your **utilization**, and it is bad. Explain in one
sentence where the waste went.

> *(your answer)*

---

## Exercise 3 — pipelining

Now feed a **second** dot product immediately behind the first:
`x² = [1, 1, 1, 1]`, injected starting one cycle after the first.

| cycle | act PE0 | act PE1 | act PE2 | act PE3 | psum PE0 | psum PE1 | psum PE2 | psum PE3 | leaves array |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0 | 0 | 0 |  |  |  |  |  |
| 1 | 1 | 6 | 0 | 0 |  |  |  |  |  |
| 2 | 0 | 1 | 7 | 0 |  |  |  |  |  |
| 3 | 0 | 0 | 1 | 8 |  |  |  |  |  |
| 4 | 0 | 0 | 0 | 1 |  |  |  |  |  |

**Q3.1** Both results should emerge. Which values, on which cycles?

> *(your answer)*

**Q3.2** One dot product took 4 cycles. Two took ____ cycles. Not 8. Why not?

> *(your answer)*

**Q3.3** Write the formula for M dot products on an N-PE array.

> *(your answer)*

**Q3.4** Recompute utilization for the two-vector case. Compare it with
Q2.2. What does this tell you about the array size a given workload wants?

> *(your answer)*

---

## Exercise 4 — the one that matters

Suppose you implement `step()` like this:

```python
for i, pe in enumerate(self.pes):
    psum_in = 0 if i == 0 else self.pes[i-1].psum_out
    pe.commit(*pe.compute(act_inputs[i], psum_in))
```

**Q4.1** Trace cycle 0 of Exercise 2 under this implementation. What is
`psum_out` for all four PEs at the end of cycle 0?

> *(your answer)*

**Q4.2** What physically impossible thing has happened?

> *(your answer)*

**Q4.3** This is the most common bug in the project, and the reason the
starter code splits `compute()` from `commit()`. Describe the rule in one
sentence.

> *(your answer)*

---

## Submit

Push this completed file to `submissions/<your-roll>/week1/worksheet.md`.

If you're unsure about an answer, write what you think and say you're unsure.
A wrong answer you reasoned toward is worth full marks here; a blank is not.
