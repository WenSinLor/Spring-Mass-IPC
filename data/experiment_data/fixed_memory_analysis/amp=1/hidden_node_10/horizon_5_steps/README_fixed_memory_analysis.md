# Fixed Delayed-Input Linear Memory Analysis

This analysis should be used for topology-memory comparison.

Do not compare topology-specific selected H,D from hidden-node basis selection as
the main memory evidence. The hidden-node target changes with topology, so a
topology-specific basis selection does not define a common task demand.

This script measures fixed delayed-input recall:

    s_m^(theta) -> u_{m-tau}

The delayed-input target is common across topologies, so it is a fair memory
benchmark. Actual hidden-node task performance should be measured separately:

    s_m^(theta) -> y_{m+h}^{(theta)}

The comparison script merges these fixed memory metrics with actual hidden-node
prediction NMSE. A positive relationship supports, but does not prove, that
memory-enhanced visible-state encoding contributes to hidden-node prediction.
