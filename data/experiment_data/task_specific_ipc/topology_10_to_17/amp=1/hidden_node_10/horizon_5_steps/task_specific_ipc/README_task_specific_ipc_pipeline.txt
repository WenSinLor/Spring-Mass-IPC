# Task-Specific IPC Pipeline

This output uses one frozen input-history Legendre dictionary selected across a
topology family. The hidden-node target changes with topology, so the common
dictionary keeps the task-weighted IPC scores comparable.

H_cut and D_cut are the truncation of the input-history polynomial dictionary.
They are not a direct measurement of the exact memory depth or nonlinearity of
any individual topology.

c_alpha measures how strongly the topology-conditioned hidden-node target uses
basis term alpha. IPC_alpha measures how well the visible reservoir state
linearly reconstructs that same basis term. The task-specific score is the
demand-supply match:

    Q_twIPC = sum_alpha c_alpha * IPC_alpha

If the basis adequacy label is weak, Q_twIPC is only a partial input-driven
explanation of hidden-node prediction performance.
