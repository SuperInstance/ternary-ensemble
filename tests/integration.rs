//! Tests for ternary-ensemble.

use ternary_ensemble::*;

fn make_sample(features: Vec<f64>, label: TernaryLabel) -> TernarySample {
    TernarySample::new(features, label)
}

fn make_dataset() -> Vec<TernarySample> {
    vec![
        make_sample(vec![0.1, 0.2], 0),
        make_sample(vec![0.3, 0.1], 0),
        make_sample(vec![0.2, 0.3], 0),
        make_sample(vec![0.9, 0.8], 1),
        make_sample(vec![0.8, 0.9], 1),
        make_sample(vec![0.85, 0.75], 1),
        make_sample(vec![1.5, 1.6], 2),
        make_sample(vec![1.4, 1.5], 2),
        make_sample(vec![1.6, 1.4], 2),
    ]
}

fn make_agents(n: usize) -> Vec<WeakAgent> {
    (0..n)
        .map(|i| {
            WeakAgent::with_accuracy(i, 2, 0.35 + (i as f64 * 0.04).min(0.20), i as u64 * 17 + 3)
        })
        .collect()
}

// ---- WeakAgent tests ----

#[test]
fn test_weak_agent_creation() {
    let agent = WeakAgent::new(0, vec![0.5, -0.3], 0.1, 0.45, 42);
    assert_eq!(agent.id, 0);
    assert!((agent.accuracy - 0.45).abs() < 1e-10);
}

#[test]
fn test_weak_agent_accuracy_bounds() {
    let _ok = WeakAgent::new(0, vec![1.0], 0.0, 0.59, 0);
    let result = std::panic::catch_unwind(|| {
        WeakAgent::new(0, vec![1.0], 0.0, 0.60, 0);
    });
    assert!(result.is_err());
}

#[test]
fn test_weak_agent_with_accuracy() {
    let agent = WeakAgent::with_accuracy(1, 3, 0.50, 99);
    assert_eq!(agent.id, 1);
    assert!((agent.accuracy - 0.50).abs() < 1e-10);
    assert_eq!(agent.weights.len(), 3);
}

#[test]
fn test_weak_agent_predict() {
    let agent = WeakAgent::new(0, vec![1.0, 1.0], 0.0, 0.4, 0);
    let sample = make_sample(vec![0.1, 0.1], 0);
    let pred = agent.predict(&sample);
    assert!(pred <= 2);
}

#[test]
fn test_weak_agent_predict_batch() {
    let agent = WeakAgent::new(0, vec![1.0, -1.0], 0.5, 0.5, 0);
    let samples = make_dataset();
    let preds = agent.predict_batch(&samples);
    assert_eq!(preds.len(), samples.len());
    for p in &preds {
        assert!(*p <= 2);
    }
}

#[test]
fn test_weak_agent_predict_with_confidence() {
    let agent = WeakAgent::new(0, vec![1.0, 1.0], 0.0, 0.5, 0);
    let sample = make_sample(vec![1.0, 1.0], 2);
    let (label, conf) = agent.predict_with_confidence(&sample);
    assert!(label <= 2);
    assert!((0.0..=1.0).contains(&conf));
}

// ---- TernarySample tests ----

#[test]
fn test_ternary_sample_valid_labels() {
    for label in 0u8..=2 {
        let s = TernarySample::new(vec![1.0], label);
        assert_eq!(s.label, label);
    }
}

#[test]
fn test_ternary_sample_invalid_label() {
    let result = std::panic::catch_unwind(|| {
        TernarySample::new(vec![1.0], 3);
    });
    assert!(result.is_err());
}

// ---- Voting tests ----

#[test]
fn test_majority_vote() {
    let agents = make_agents(5);
    let combiner = VotingCombiner::new(VotingStrategy::Majority);
    let sample = make_sample(vec![0.5, 0.5], 1);
    let pred = combiner.predict(&agents, &sample);
    assert!(pred <= 2);
}

#[test]
fn test_weighted_vote() {
    let agents = make_agents(5);
    let weights = vec![0.3, 0.1, 0.2, 0.15, 0.25];
    let combiner = VotingCombiner::weighted(weights);
    let sample = make_sample(vec![0.5, 0.5], 1);
    let pred = combiner.predict(&agents, &sample);
    assert!(pred <= 2);
}

#[test]
fn test_ranked_vote() {
    let agents = make_agents(5);
    let combiner = VotingCombiner::new(VotingStrategy::Ranked);
    let sample = make_sample(vec![0.5, 0.5], 1);
    let pred = combiner.predict(&agents, &sample);
    assert!(pred <= 2);
}

// ---- Ensemble tests ----

#[test]
fn test_ensemble_voting_majority() {
    let agents = make_agents(5);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
    let ensemble = Ensemble::new(agents, strategy);
    assert_eq!(ensemble.len(), 5);
    let sample = make_sample(vec![0.5, 0.5], 1);
    let pred = ensemble.predict(&sample);
    assert!(pred <= 2);
}

#[test]
fn test_ensemble_predict_batch() {
    let agents = make_agents(3);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Weighted));
    let ensemble = Ensemble::new(agents, strategy);
    let samples = make_dataset();
    let preds = ensemble.predict_batch(&samples);
    assert_eq!(preds.len(), samples.len());
}

#[test]
fn test_ensemble_empty_panics() {
    let result = std::panic::catch_unwind(|| {
        let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
        Ensemble::new(vec![], strategy);
    });
    assert!(result.is_err());
}

// ---- Boosting tests ----

#[test]
fn test_boosting_combiner_fit_and_predict() {
    let agents = make_agents(5);
    let samples = make_dataset();
    let mut booster = BoostingCombiner::new(5, 0.1);
    booster.fit(&agents, &samples);
    assert_eq!(booster.agent_weights.len(), 5);
    let pred = booster.predict(&agents, &samples[0]);
    assert!(pred <= 2);
}

#[test]
fn test_boosting_fit_empty_agents_panics() {
    let result = std::panic::catch_unwind(|| {
        let mut booster = BoostingCombiner::new(3, 0.1);
        booster.fit(&[], &make_dataset());
    });
    assert!(result.is_err());
}

#[test]
fn test_boosting_fit_empty_samples_panics() {
    let result = std::panic::catch_unwind(|| {
        let agents = make_agents(3);
        let mut booster = BoostingCombiner::new(3, 0.1);
        booster.fit(&agents, &[]);
    });
    assert!(result.is_err());
}

#[test]
fn test_boosting_ensemble() {
    let agents = make_agents(5);
    let samples = make_dataset();
    let mut booster = BoostingCombiner::new(10, 0.1);
    booster.fit(&agents, &samples);
    let strategy = CombineStrategy::Boosting(booster);
    let ensemble = Ensemble::new(agents, strategy);
    let preds = ensemble.predict_batch(&samples);
    assert_eq!(preds.len(), samples.len());
}

// ---- Stacking tests ----

#[test]
fn test_stacking_unfitted_falls_back_to_majority() {
    let agents = make_agents(3);
    let stacker = StackingCombiner::default_params();
    assert!(!stacker.fitted);
    let sample = make_sample(vec![0.5, 0.5], 1);
    let pred = stacker.predict(&agents, &sample);
    assert!(pred <= 2);
}

#[test]
fn test_stacking_fit_and_predict() {
    let agents = make_agents(5);
    let samples = make_dataset();
    let mut stacker = StackingCombiner::new(0.01, 50);
    stacker.fit(&agents, &samples);
    assert!(stacker.fitted);
    let pred = stacker.predict(&agents, &samples[0]);
    assert!(pred <= 2);
}

#[test]
fn test_stacking_fit_empty_agents_panics() {
    let result = std::panic::catch_unwind(|| {
        let mut stacker = StackingCombiner::new(0.01, 10);
        stacker.fit(&[], &make_dataset());
    });
    assert!(result.is_err());
}

#[test]
fn test_stacking_fit_empty_samples_panics() {
    // Regression: previously this did NOT panic, but silently produced
    // meta_bias = [NaN, NaN, NaN] with fitted=true, after which every
    // predict() returned 2 regardless of input. Now it must panic.
    let result = std::panic::catch_unwind(|| {
        let agents = make_agents(3);
        let mut stacker = StackingCombiner::new(0.01, 10);
        stacker.fit(&agents, &[]);
    });
    assert!(result.is_err());
}

#[test]
fn test_stacking_ensemble() {
    let agents = make_agents(5);
    let samples = make_dataset();
    let mut stacker = StackingCombiner::new(0.01, 50);
    stacker.fit(&agents, &samples);
    let strategy = CombineStrategy::Stacking(stacker);
    let ensemble = Ensemble::new(agents, strategy);
    let acc = ensemble.evaluator().accuracy(&samples);
    assert!((0.0..=1.0).contains(&acc));
}

// ---- Evaluator tests ----

#[test]
fn test_evaluator_accuracy() {
    let agents = make_agents(7);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
    let ensemble = Ensemble::new(agents, strategy);
    let acc = ensemble.evaluator().accuracy(&make_dataset());
    assert!((0.0..=1.0).contains(&acc));
}

#[test]
fn test_evaluator_detailed() {
    let agents = make_agents(7);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Weighted));
    let ensemble = Ensemble::new(agents, strategy);
    let result = ensemble.evaluator().evaluate(&make_dataset());
    assert_eq!(result.n_samples, 9);
    assert_eq!(result.individual_accuracies.len(), 7);
    assert!(result.ensemble_accuracy >= 0.0 && result.ensemble_accuracy <= 1.0);
}

#[test]
fn test_evaluator_improvement() {
    let agents = make_agents(7);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
    let ensemble = Ensemble::new(agents, strategy);
    let result = ensemble.evaluator().evaluate(&make_dataset());
    // Improvement values are valid (could be negative if ensemble is worse)
    let _imp_best = result.improvement_over_best();
    let _imp_avg = result.improvement_over_average();
}

#[test]
fn test_evaluator_precision_recall() {
    let agents = make_agents(7);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
    let ensemble = Ensemble::new(agents, strategy);
    let result = ensemble.evaluator().evaluate(&make_dataset());
    let precision = result.precision();
    let recall = result.recall();
    for c in 0..3 {
        assert!(precision[c] >= 0.0 && precision[c] <= 1.0);
        assert!(recall[c] >= 0.0 && recall[c] <= 1.0);
    }
}

#[test]
fn test_evaluator_compare() {
    let agents = make_agents(5);
    let strategy = CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
    let ensemble = Ensemble::new(agents, strategy);
    let comparisons = ensemble.evaluator().compare(&make_dataset());
    assert_eq!(comparisons.len(), 5);
    for (i, agent_acc, ens_acc) in &comparisons {
        assert!(*i < 5);
        assert!((*agent_acc >= 0.0) && (*agent_acc <= 1.0));
        assert!((*ens_acc >= 0.0) && (*ens_acc <= 1.0));
    }
}
