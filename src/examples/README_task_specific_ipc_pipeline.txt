Task-specific IPC pipeline
==========================

This pipeline compares a topology family using one common input-history
Legendre dictionary.

Why one common dictionary?
The hidden-node target changes with topology. If each topology selects its own
final H,D, the task weights are not defined on the same basis and are not
directly comparable. The selection script therefore scans topology_10_prestress
through topology_17_prestress together, chooses one common H_cut,D_cut, and
freezes that dictionary for all later scores.

What H_cut,D_cut mean
H_cut and D_cut are the fixed truncation of the input-history polynomial
dictionary. They are not the exact memory depth or nonlinearity of a topology.
For horizon h, readout-time delays [0,...,H_cut] correspond to target-time
delays [h,...,h+H_cut].

What task weights mean
c_alpha measures how much the topology-conditioned hidden-node target depends
on basis term alpha in the measured input-history dictionary.

What IPC_alpha means
IPC_alpha measures how well the visible reservoir state linearly reconstructs
basis term alpha.

What Q_twIPC means
Q_twIPC is a demand-supply match:

    Q_twIPC = sum_alpha c_alpha * IPC_alpha

Topology comparison
If topology 17 has higher Q_twIPC and lower actual hidden-node NMSE than
topology 10, then topology 17 supplies more of the input-history components
that are relevant to its hidden-node task under the common dictionary.

Limitation
If basis adequacy is weak, Q_twIPC is a partial input-driven explanation, not
an exact predictor of the full hidden-node NMSE.

Typical commands
----------------

python src/examples/select_common_task_dictionary.py --topology-start 10 --topology-stop 17 --amplitude amp=1 --hidden-node 10 --reference-node 0 --horizon-steps 5 --H-max 30 --D-max 2 --sample all

python src/examples/compute_task_specific_ipc.py --selected-dictionary path/to/selected_common_dictionary.json --topology-start 10 --topology-stop 17 --amplitude amp=1 --hidden-node 10 --reference-node 0 --horizon-steps 5 --sample all

python src/examples/compare_twipc_with_hidden_nmse.py --twipc-csv path/to/task_specific_ipc_per_sample.csv --hidden-metrics-csv path/to/hidden_node_prediction_metrics.csv
