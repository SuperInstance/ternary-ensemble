# Ternary Ensemble — Combine Weak Ternary Agents into Strong Predictors

**Ternary Ensemble** provides ensemble methods for agents whose outputs are ternary labels {0, 1, 2} (mapped to {-1, 0, +1}). It implements majority voting, AdaBoost-style boosting, and stacking (meta-learning) — the three foundational ensemble strategies adapted for three-class prediction where the neutral class plays a special role in confidence-weighted decisions.

## Why It Matters

Individual ternary agents are weak learners — each one makes three-way decisions with limited accuracy. But collectively, their errors are largely independent, so combining many weak agents produces a strong predictor. This is the Condorcet Jury Theorem in action: if each agent has accuracy p > ⅓ (better than random for 3 classes), then N agents voting together have accuracy approaching 1 as N grows. The boosting strategy goes further: it trains each subsequent agent to focus on the samples previous agents got wrong, iteratively reducing error. For ternary fleet systems, ensembles provide the reliability that individual GPU agents cannot achieve alone.

## How It Works

### Voting

The `VotingCombiner` supports three strategies:
- **Majority**: The label with the most votes wins. O(N·K) for N agents, K samples.
- **Weighted**: Each agent's vote is weighted by its accuracy on a validation set. Higher-accuracy agents have more influence.
- **Plurality**: The label with the most votes wins, even without a strict majority (useful when votes split three ways).

### Boosting

The `BoostingCombiner` implements ternary AdaBoost:
1. Initialize sample weights uniformly: w_i = 1/K
2. Train weak agent on weighted samples
3. Compute weighted error: ε = Σ w_i · [agent wrong]
4. Agent weight: α = ½ · ln((1-ε)/ε)
5. Update sample weights: increase for misclassified, decrease for correct
6. Repeat for T rounds

Final prediction: weighted majority vote of all T agents. Training is O(T · N · K).

### Stacking

The `StackingCombiner` trains a meta-learner on the outputs of base agents. Level-1 features are the ternary predictions of each base agent; the meta-learner learns which agent to trust for which input pattern. This captures complementary strengths: agent A may be good at distinguishing class 0 from class 1, while agent B excels at 1 vs 2.

### Evaluation

The `EnsembleEvaluator` provides accuracy, per-class precision/recall/F1, and a confusion matrix — all computed in O(K) per evaluation pass.

## Quick Start

```rust
use ternary_ensemble::{Ensemble, VotingCombiner, VotingStrategy, WeakAgent, TernarySample};

// Create training data
let samples = vec![
    TernarySample::new(vec![1.0, 0.0], 0),
    TernarySample::new(vec![0.0, 1.0], 1),
    TernarySample::new(vec![1.0, 1.0], 2),
];

// Train weak agents and combine
let mut ensemble = Ensemble::new(Box::new(VotingCombiner::new(VotingStrategy::Majority)));
ensemble.add_agent(WeakAgent::train(&samples));
ensemble.add_agent(WeakAgent::train(&samples));

let prediction = ensemble.predict(&[1.0, 0.0]);
```

```bash
cargo add ternary-ensemble
```

## API

| Type / Function | Description |
|---|---|
| `WeakAgent` | Individual ternary classifier |
| `Ensemble` | Manages agents + combination strategy |
| `VotingCombiner` | Majority/weighted/plurality voting |
| `BoostingCombiner` | Ternary AdaBoost |
| `StackingCombiner` | Meta-learner over base agents |
| `EnsembleEvaluator` | Accuracy, precision/recall, confusion matrix |

## Architecture Notes

Ensembles are the reliability mechanism in **SuperInstance**: fleet decisions are made by ensembles of ternary agents, not single agents. The γ + η = C conservation law applies to ensemble diversity: too much agreement (low η) means the ensemble is redundant; too much disagreement (high η) means predictions are unreliable. Optimal performance balances γ (collective signal) and η (agent diversity). See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Schapire, Robert. "The Strength of Weak Learnability," *Machine Learning*, 5(2), 1990 — boosting theory.
- Breiman, Leo. "Random Forests," *Machine Learning*, 45(1), 2001 — ensemble methods.
- Wolpert, David. "Stacked Generalization," *Neural Networks*, 5(2), 1992 — stacking.

## License

MIT
