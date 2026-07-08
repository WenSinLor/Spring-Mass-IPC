# X-Only Task-Specific IPC

This output compares task-specific IPC for the hidden-node x target only.
The hidden-node y target is intentionally excluded because its input-history
Legendre expansion was not accurate enough for the current data.

For each sample, the script:

1. Builds one common Legendre input-history dictionary.
2. Fits the x target as a centered expansion of that dictionary.
3. Computes task weights using c_alpha_x = a_alpha_x^2 ||P_alpha||^2 / ||x||^2.
4. Computes IPC_alpha by asking how well visible reservoir states reconstruct
   each basis term.
5. Computes Q_x_twIPC = sum_alpha c_alpha_x * IPC_alpha.
6. Computes the actual hidden-node readout NMSE_x for comparison.

Topology-level values are means and standard deviations across samples, not
one pooled fit across all samples.

The task weights are not renormalized. With strongly correlated measured input,
the finite Legendre basis terms are not orthogonal in the sampled data, so
sum_alpha c_alpha_x can exceed 1. This is a diagnostic warning from the data,
not a post-processing error.
