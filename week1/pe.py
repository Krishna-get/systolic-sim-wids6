"""Week 1 — the processing element.

A single weight-stationary multiply-accumulate unit. This is the entire
compute hardware of a systolic array; everything else is wiring.

Fill in the TODOs. Read week1/README.md §2 and §3 first, and fill in
worksheet.md before you write any code here.
"""


class ProcessingElement:
    """One weight-stationary MAC unit.

    Every cycle this PE:
      1. takes an activation (from its left neighbour, or injected directly)
      2. takes a partial sum (from the PE above / before it in the chain)
      3. computes  psum_out = psum_in + weight * act_in
      4. passes the activation onward, unchanged
      5. passes the new partial sum onward

    THE CRITICAL DETAIL: both outputs are REGISTERED. They do not appear on
    the same cycle the inputs arrived — they become visible on the NEXT
    cycle. That one-cycle delay per hop is what creates the wavefront, and
    forgetting it is the single most common Week 1 bug.

    This is why compute() and commit() are separate methods. compute() must
    be a pure function of its arguments; it must NOT touch self. All PEs in
    the array compute from the OLD state, and only then does everyone commit.
    See week1/README.md §3, "the golden rule".
    """

    def __init__(self, weight=0):
        self.weight = weight
        self.act_out = 0    # registered output, visible to the next PE
        self.psum_out = 0   # registered output, visible to the next PE

    def load_weight(self, w):
        """Weight-stationary: set once, held for the whole tile.

        In Week 2 this will cost real cycles. For now it is free.
        """
        # TODO
        raise NotImplementedError

    def compute(self, act_in, psum_in):
        """Return the (act_out, psum_out) this PE WOULD latch this cycle.

        Must be pure — do not assign to self here. The array calls compute()
        on every PE before calling commit() on any of them.

        Returns:
            tuple[int, int]: (act_out, psum_out)
        """
        # TODO
        raise NotImplementedError

    def commit(self, act_out, psum_out):
        """Latch the values returned by compute(). Called once per cycle,
        after every PE in the array has computed."""
        # TODO
        raise NotImplementedError

    def reset(self):
        """Clear the output registers. Keeps the weight — it's stationary."""
        self.act_out = 0
        self.psum_out = 0

    def __repr__(self):
        return (f"PE(w={self.weight}, act_out={self.act_out}, "
                f"psum_out={self.psum_out})")
