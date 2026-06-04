# ternary-ensemble

Ensemble methods for ternary agents — combine multiple weak agents into a strong one.

## Overview

This crate implements ensemble strategies for agents that produce **ternary** (3-class) outputs (labels `0`, `1`, `2`). Each individual "weak agent" has accuracy below 60%, but by combining many of them, the ensemble can achieve significantly higher accuracy.

## Core Concepts

### WeakAgent

A simple agent with known accuracy < 60%. It uses a linear threshold model over feature vectors to produce ternary predictions.

```rust
use ternary_ensemble::WeakAgent;

let agent = WeakAgent::with_accuracy(0, 2, 0.45, 42);
let sample = ternary_ensemble::TernarySample::new(vec![0.5, 0.3], 1);
let prediction = agent.predict(&sample);
```

### Ensemble Strategies

#### 1. Voting (Majority / Weighted / Ranked)

The simplest ensemble: combine predictions by vote.

- **Majority vote** — each agent gets one equal vote
- **Weighted vote** — agent votes are weighted by accuracy or custom weights
- **Ranked vote** — Borda count across confidence rankings

```rust
use ternary_ensemble::*;

let agents = vec![
    WeakAgent::with_accuracy(0, 2, 0.40, 1),
    WeakAgent::with_accuracy(1, 2, 0.45, 2),
    WeakAgent::with_accuracy(2, 2, 0.50, 3),
];

let ensemble = Ensemble::new(
    agents,
    CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority)),
);
```

#### 2. Boosting (AdaBoost-style)

Sequential reweighting: misclassified samples get higher weight, agents are scored by weighted accuracy.

```rust
let mut booster = BoostingCombiner::new(10, 0.1);
booster.fit(&agents, &training_data);
let ensemble = Ensemble::new(agents, CombineStrategy::Boosting(booster));
```

#### 3. Stacking (Meta-Learner)

A meta-learner trains on the outputs of base agents, learning optimal combination weights via gradient descent.

```rust
let mut stacker = StackingCombiner::new(0.01, 100);
stacker.fit(&agents, &training_data);
let ensemble = Ensemble::new(agents, CombineStrategy::Stacking(stacker));
```

### Evaluation

```rust
let evaluator = ensemble.evaluator();
let result = evaluator.evaluate(&test_data);

println!("Ensemble accuracy: {:.2}%", result.ensemble_accuracy * 100.0);
println!("Improvement over best individual: {:.2}%", result.improvement_over_best() * 100.0);
println!("Precision: {:?}", result.precision());
println!("Recall: {:?}", result.recall());
```

## Why Ternary?

Ternary classification (3 classes) is a natural fit for many real-world scenarios:
- Sentiment: negative / neutral / positive
- Trend: down / flat / up
- Decision: reject / abstain / accept

Ensemble methods are particularly effective here because the decision boundaries are more nuanced than binary, making weak agents common but strong combinations achievable.

## Features

- **Pure Rust** — no unsafe code, no external dependencies
- **Three combination strategies**: Voting, Boosting, Stacking
- **Full evaluation suite**: accuracy, precision, recall, confusion matrix, improvement metrics
- **24 tests** covering all major functionality

## License

MIT
