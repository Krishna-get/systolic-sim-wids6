"""Week 1 — a 1D systolic chain.

N processing elements chained together, computing a dot product.

STRUCTURE (read this carefully — it is the thing the worksheet makes you
internalise, and it is exactly one column of the 2D array you build in
Week 2):

    PE i permanently holds weight w[i].

    Activations are injected into each PE directly — x[i] goes to PE i.
    (In the 2D array these arrive from the neighbouring column; in 1D we
    inject them from outside.)

    Partial sums CHAIN along the array: PE i's psum_in is PE (i-1)'s
    psum_out from the previous cycle. PE 0's psum_in is always 0.

    The completed dot product emerges from PE N-1.

THE STAGGER: because the partial sum takes one cycle per hop, x[i] must be
injected into PE i at cycle i, not all at once at cycle 0. The partial sum
wavefront travels down the chain and has to meet each activation exactly
when it arrives. Inject everything at cycle 0 and you will get a wrong
answer; inject with the wrong stagger and you will get a different wrong
answer. Work the timing out on paper first.

Expected result for N=4, weights [1,2,3,4], activations [5,6,7,8]:
the value 70 emerges at the end of cycle 3, i.e. after 4 cycles.
"""

from pe import ProcessingElement


class SystolicRow:
    """A chain of N weight-stationary PEs computing dot products."""

    def __init__(self, n_pes):
        self.n_pes = n_pes
        self.pes = [ProcessingElement() for _ in range(n_pes)]
        self.cycle = 0
        self.trace = []     # one entry per cycle, for print_trace()

    # ------------------------------------------------------------------

    def load_weights(self, weights):
        """Load one weight per PE. len(weights) must equal n_pes."""
        # TODO
        raise NotImplementedError

    def step(self, act_inputs):
        """Advance the array by exactly one cycle.

        Args:
            act_inputs: list of length n_pes. act_inputs[i] is the activation
                presented to PE i on this cycle. Use 0 where no activation is
                arriving.

        Returns:
            int: the partial sum leaving PE n_pes-1 at the END of this cycle,
                i.e. the value it has just latched.

        Implementation order matters. You must:
            1. collect psum_in for every PE from the CURRENT (old) state
            2. call compute() on every PE
            3. only then call commit() on every PE

        If you compute and commit PE by PE in a single loop, data will skip
        cells and your cycle counts will be silently too low. Your dot
        product will still come out right, which is exactly what makes this
        bug so dangerous.
        """
        # TODO
        raise NotImplementedError

    def run(self, activation_vectors):
        """Run one or more dot products through the array.

        Args:
            activation_vectors: list of vectors, each of length n_pes.
                Vector m is injected starting at cycle m — consecutive dot
                products are pipelined one cycle apart.

        Returns:
            tuple[list[int], int]: (results, total_cycles)

        For a single vector on an N-PE array this should take N cycles.
        For M vectors it should take N + M - 1 — think about why before you
        check.
        """
        # TODO
        raise NotImplementedError

    def run_single(self, activations):
        """Convenience wrapper: one dot product. Returns (result, cycles)."""
        results, cycles = self.run([activations])
        return results[0], cycles

    # ------------------------------------------------------------------

    def reset(self):
        for pe in self.pes:
            pe.reset()
        self.cycle = 0
        self.trace = []

    def print_trace(self):
        """Pretty-print the cycle-by-cycle table.

        Write this early and make the columns line up. You will stare at its
        output more than at any other thing this month, and in Week 2 you
        will be very glad it already exists.

        Suggested format:

            cyc |  act_in           | psum_out          | out
            ----+-------------------+-------------------+-----
              0 |  5    0    0    0 |  5    0    0    0 |   0
              1 |  0    6    0    0 |  0   17    0    0 |   0
              2 |  0    0    7    0 |  0    0   38    0 |   0
              3 |  0    0    0    8 |  0    0    0   70 |  70
        """
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    row = SystolicRow(4)
    row.load_weights([1, 2, 3, 4])
    result, cycles = row.run_single([5, 6, 7, 8])
    row.print_trace()
    print(f"\nresult = {result} (expected 70), cycles = {cycles} (expected 4)")
