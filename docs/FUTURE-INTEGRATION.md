# Future Integration: ternary-ensemble

## Current State
Provides ensemble methods for combining ternary agents: `Ensemble` manages multiple `WeakAgent` instances, `EnsembleCombiner` strategies (majority vote, weighted vote), `StackingCombiner` meta-learner, and `EvaluationResult` with confusion matrix, per-class precision/recall, and improvement-over-baseline metrics.

## Integration Opportunities

### With ternary-cell (Ensemble Cell Grids)
Instead of a single cell grid, run an ensemble of grids with different initial conditions, different noise seeds, or different neighborhood structures. Each grid is a `WeakAgent`. `EnsembleCombiner::WeightedVote` combines their outputs — grids with higher recent accuracy get more weight. `EvaluationResult::improvement_over_best()` quantifies whether the ensemble outperforms the best individual grid. This is the cell-grid equivalent of random forests.

### With ternary-federated (Federated Ensemble)
ternary-federated aggregates strategies across distributed rooms. ternary-ensemble combines multiple agents' outputs. Together: each federated node runs a local ensemble. Federated aggregation combines the ensemble combiners (not individual agents) — meta-learning across distributed ensembles. `StackingCombiner`'s meta-weights are aggregated via federated rounds, sharing learned combination strategies without sharing raw data.

### With ternary-adversarial (Adversarial Ensemble)
Single strategies are vulnerable to adversarial attack. Ensembles are harder to attack because the adversary must fool a majority of agents simultaneously. `DefenseReport` should test against ensemble targets, not individual agents. `StackingCombiner` can learn to weigh agents by their adversarial robustness — robust agents get more weight when under attack.

## Potential in Mature Systems
In room-as-codespace, PLATO runs ensemble rooms — multiple Codespaces processing the same task with different configurations. The ensemble combiner produces the final output. This provides fault tolerance (individual Codespace failures don't crash the system), robustness (adversarial inputs must fool multiple rooms), and accuracy (ensemble improvement over individuals). For ESP32, the ensemble is micro-scale: 3 tiny agents in the lookup table, combined by majority vote in <1µs.

## Cross-Pollination Ideas
- **ternary-diversity**: Measure ensemble diversity using entropy — diverse ensembles are more robust. Prune agents that are too similar to others.
- **ternary-transfer**: Transfer ensemble combiner weights between rooms — a stacking combiner trained in Room A may work well in similar Room B.
- **ternary-pareto**: Pareto-optimal ensembles — select the ensemble size and combiner that maximizes accuracy while minimizing resource usage.

## Dependencies for Next Steps
- Define `CellGridEnsemble` wrapping ternary-cell grids as `WeakAgent` instances
- Add ensemble evaluation to ternary-cell's conservation phase (compare ensemble vs. individual)
- Implement federated ensemble aggregation combining stacking combiners
- Benchmark micro-ensemble (3 agents) on ESP32 for real-time majority vote
