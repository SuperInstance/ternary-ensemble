# Ternary Ensemble — Combine Weak Ternary Agents into Strong Predictors

**Ternary Ensemble** provides ensemble methods for agents whose outputs are ternary labels {0, 1, 2} (mapped to {-1, 0, +1}). It implements majority voting, AdaBoost-style boosting, and stacking (meta-learning) — the three foundational ensemble strategies adapted for three-class prediction where the neutral class plays a special role in confidence-weighted decisions.

## Why It Matters

Individual ternary agents are weak learners — each one makes three-way decisions with limited accuracy. But collectively, their errors are largely independent, so combining many weak agents produces a strong predictor. This is the Condorcet Jury Theorem in action: if each agent has accuracy p > ⅓ (better than random for 3 classes), then N agents voting together have accuracy approaching 1 as N grows. The boosting strategy goes further: it trains each subsequent agent to focus on the samples previous agents got wrong, iteratively reducing error. For ternary fleet systems, ensembles provide the reliability that individual GPU agents cannot achieve alone.

## How It Works

### Voting

The `VotingCombiner` supports three strategies ([`VotingStrategy`](src/voting.rs)):
- **Majority**: Each agent casts one equal vote; the label with the most votes wins.
- **Weighted**: Each agent's vote is multiplied by a per-agent weight. If the supplied `agent_weights` vector is shorter than the agent list, the missing entries silently fall back to each agent's `accuracy` field.
- **Ranked**: Each agent assigns Borda-count ranks (0 = best) to all three classes based on its confidence score; the class with the lowest total rank wins.

Tie-breaking is deterministic and uniform across all three strategies: **the lowest class index wins**. So a 1-1-1 tie resolves to 0, a tie between 1 and 2 resolves to 1, etc.

### Boosting

The `BoostingCombiner` implements ternary AdaBoost:

1. Initialize sample weights uniformly: `w_i = 1/K`
2. For each round `t = 1..T`:
    1. Compute each agent's weighted error: `ε_a = Σ_i w_i · [agent_a wrong on sample_i]`
    2. Update each agent's weight: `α_a = ½ · ln((1 - ε_a) / ε_a) · learning_rate`, then `agent_weights[a] += α_a` and renormalize so the weights sum to 1.
    3. Compute the ensemble's weighted-vote prediction for every sample.
    4. Update sample weights: multiply by `(1 + learning_rate)` if the ensemble was wrong, or `(1 - learning_rate / 2)` if it was right, then renormalize.

Final prediction: weighted majority vote of all agents using the learned `agent_weights`. Training is `O(T · N · K)` for `T` rounds, `N` agents, `K` samples.

### Stacking

The `StackingCombiner` trains a per-class meta-learner on the outputs of the base agents. For each sample, the meta-features are the base agents' predictions (one-hot encoded); the meta-learner learns a `[n_agents × 3]` weight matrix plus a 3-element bias that maps those one-hots to per-class logits, trained by gradient descent on a hinge-like loss. The predicted class is the argmax of the resulting logits.

If `predict` is called before `fit`, the combiner silently falls back to simple majority voting — the `fitted` field is the explicit indicator of whether the meta-learner has been trained.

### Evaluation

The `EnsembleEvaluator` provides accuracy, per-class precision/recall/F1, and a confusion matrix — all computed in O(K) per evaluation pass.

## Quick Start

```rust
use ternary_ensemble::{
    CombineStrategy, Ensemble, TernarySample, VotingCombiner, VotingStrategy, WeakAgent,
};

// Two features per sample, ternary labels in {0, 1, 2}.
let samples = vec![
    TernarySample::new(vec![0.1, 0.2], 0),
    TernarySample::new(vec![0.9, 0.8], 1),
    TernarySample::new(vec![1.5, 1.6], 2),
];

// Build a few weak agents. WeakAgent::with_accuracy constructs an agent with
// pseudo-random weights and a known accuracy (must be < 0.60).
let agents: Vec<WeakAgent> = (0..5)
    .map(|i| WeakAgent::with_accuracy(i, 2, 0.40, i as u64 * 17 + 3))
    .collect();

// Combine them with majority voting.
let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
let ensemble = Ensemble::new(agents, strategy);

// Predict on a single sample (predict takes a &TernarySample, not a &[f64]).
let probe = TernarySample::new(vec![0.1, 0.2], 0);
let prediction = ensemble.predict(&probe);
assert!(prediction <= 2);

// Evaluate accuracy / per-class precision / confusion matrix on a test set.
let result = ensemble.evaluator().evaluate(&samples);
println!("ensemble accuracy: {:.3}", result.ensemble_accuracy);
```

```bash
cargo add ternary-ensemble
```

## API

| Type / Function | Description |
|---|---|
| `WeakAgent` | Individual ternary classifier with `predict`, `predict_batch`, `predict_with_confidence` |
| `TernarySample` | Feature vector + ternary label in `{0, 1, 2}` |
| `Ensemble` | Holds `Vec<WeakAgent>` + a `CombineStrategy`; entry point for `predict` / `predict_batch` / `evaluator` |
| `CombineStrategy` | Enum: `Voting(VotingCombiner)`, `Boosting(BoostingCombiner)`, `Stacking(StackingCombiner)` |
| `VotingCombiner` / `VotingStrategy` | Majority / Weighted / Ranked (Borda) voting |
| `BoostingCombiner` | Ternary AdaBoost — call `fit` before `predict` |
| `StackingCombiner` | Meta-learner over base agents — call `fit` before `predict` |
| `EnsembleEvaluator` / `EvaluationResult` | Accuracy, per-class precision/recall, confusion matrix |

## Architecture Notes

Ensembles are the reliability mechanism in **SuperInstance**: fleet decisions are made by ensembles of ternary agents, not single agents. The γ + η = C conservation law applies to ensemble diversity: too much agreement (low η) means the ensemble is redundant; too much disagreement (high η) means predictions are unreliable. Optimal performance balances γ (collective signal) and η (agent diversity). See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Schapire, Robert. "The Strength of Weak Learnability," *Machine Learning*, 5(2), 1990 — boosting theory.
- Breiman, Leo. "Random Forests," *Machine Learning*, 45(1), 2001 — ensemble methods.
- Wolpert, David. "Stacked Generalization," *Neural Networks*, 5(2), 1992 — stacking.

## License

MIT
