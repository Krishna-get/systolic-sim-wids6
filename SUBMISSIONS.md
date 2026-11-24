# How to Submit

Read this once in Week 1. It won't change.

---

## One-time setup

```bash
git clone <repo-url>
cd systolic-sim-wids6
git checkout -b wids6/<your-roll-number>      # e.g. wids6/23b1234
mkdir -p submissions/<your-roll-number>
```

Work **only** inside `submissions/<your-roll-number>/`. Never edit files outside
it — if you think a starter file or test is wrong, open an issue rather than
changing it. (Sometimes you'll be right. Then I'll fix it for everyone.)

Your folder should end up looking like:

```
submissions/23b1234/
├── week1/
│   ├── worksheet.md
│   ├── pe.py
│   ├── array_1d.py
│   ├── trace.txt
│   └── answers.md
├── week2/
├── week3/
└── final/
```

---

## Each week

1. Copy the week's skeleton files into your folder and fill in the TODOs.
2. Run the tests until they pass:
   ```bash
   pytest week1/test_week1.py -v --impl submissions/<your-roll>/week1
   ```
3. Write `answers.md` — the written questions from that week's README.
4. Commit and push:
   ```bash
   git add submissions/<your-roll-number>
   git commit -m "week1: 1D systolic row"
   git push origin wids6/<your-roll-number>
   ```
5. Open a **pull request** against `main`, titled `[Week N] <your name>`.

Keep **one PR open per week**. I review in the PR; that's the main feedback
channel. Don't merge it yourself.

---

## Deadlines

Sunday 23:59 IST, each week. Push whatever you have.

An incomplete submission with an honest `answers.md` saying what you got stuck
on is worth far more than a silent miss, and it's how I know what to cover in
Monday's call. Several people being stuck on the same thing is useful
information, not a problem.

---

## What's graded

| | Weight | What I'm looking for |
|---|---|---|
| Tests passing | 40% | Including the golden cycle-count tests from Week 2 |
| Written answers | 35% | Reasoning, not length. Three good sentences beat a page. |
| Experiments & plots | 25% | From Week 2 onward |

The written answers carry nearly as much as the code on purpose. A simulator
that produces the right number whose author can't explain why has not finished
the exercise.

---

## The correctness gate

From Week 2 onward, your simulator must reproduce the cycle counts in
`reference/expected_outputs/` before any analysis you do counts toward the
experiment marks.

This is strict for a reason. A subtly wrong cycle-level simulator produces
plausible-looking graphs and confident wrong conclusions, and there is no way
to spot it from the plots. Matching the golden numbers is the only check that
your timing model is real.

If you think a golden number is wrong, open an issue with your reasoning. That
has happened before and it has twice turned out that the golden file was the
thing with the bug.

---

## Using AI assistants

Use them. They're good at this and pretending otherwise would be silly.

Two rules:

**You must be able to explain every line you submit.** I will ask, in PR
review, why a particular line is there. "I'm not sure, it was generated" is a
fail for that submission. This isn't a trap — it's the actual skill. Code you
can't defend is code you can't debug at 2am in Week 4.

**Don't let it write your `answers.md`.** The written answers are where the
learning gets consolidated, and an LLM-written answer about *your* simulator's
behaviour will be generic and I will be able to tell. Several of the questions
ask you to compare your own measured numbers against a prediction; there's no
way to answer those without having looked.

A good use: "explain why my cycle count is 4 lower than expected" with your
code pasted in. A bad use: "write a systolic array simulator."

---

## Collaboration

Talk to each other constantly. Debug together, compare numbers, argue about
dataflows in the group channel.

Submit your own code. If you worked closely with someone on a specific problem,
say so in your commit message or `answers.md` — that's collaboration, which is
fine, and it costs you nothing to name it.

Comparing *results* across people is actively encouraged from Week 3 onward.
If your ResNet-18 numbers disagree with someone else's by 3×, one of you has a
bug, and finding out which is faster together than alone.

---

## Getting help

**Group channel** — first stop, for anything. Ask publicly; four other people
have the same problem and nobody wants to answer it five times in DMs.

**Issues** — for anything that looks like a problem with the repo itself: a
broken test, a wrong golden number, an unclear instruction, a dead link.

**Office hours** — by appointment, for anyone genuinely blocked. Being stuck
for a day is normal and productive; it's where most of the learning is. Being
stuck for three days is not. Come and talk to me at day two.

When you ask for help, include: what you expected, what you got, and the
smallest case that shows the problem. A 4×4 array with a 4-element dot product
reproduces almost every bug in this project and is far easier to reason about
than a ResNet layer.
