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

/// Build a deterministic agent whose `predict` is constant regardless of input.
///
/// `WeakAgent::predict` maps `score = w·f + b` through:
///   score < -0.3  => label 0
///   score <  0.3  => label 1
///   else          => label 2
/// Setting weights to all-zero and a chosen bias pins `score = bias`, so the
/// agent always emits `label`. Accuracy is set to 0.0 (any value < 0.60 works);
/// the deterministic-prediction tests below do not depend on accuracy.
fn constant_agent(id: usize, label: TernaryLabel) -> WeakAgent {
    let bias = match label {
        0 => -1.0,
        1 => 0.0,
        2 => 1.0,
        _ => unreachable!("label pinned by caller to 0/1/2"),
    };
    WeakAgent::new(id, vec![0.0, 0.0], bias, 0.0, 0)
}

/// Sample with arbitrary features; label is irrelevant to constant agents.
fn any_sample() -> TernarySample {
    make_sample(vec![0.0, 0.0], 0)
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

/// Real correctness check for majority voting: with 2 agents predicting 2 and
/// 1 agent predicting 0, the result MUST be 2 (the actual majority). The
/// trivial `pred <= 2` assertion in `test_majority_vote` above cannot detect a
/// buggy implementation that always returns 0; this test can.
#[test]
fn test_majority_vote_real_majority() {
    let agents = vec![
        constant_agent(0, 2),
        constant_agent(1, 2),
        constant_agent(2, 0),
    ];
    let combiner = VotingCombiner::new(VotingStrategy::Majority);
    assert_eq!(combiner.predict(&agents, &any_sample()), 2);

    // Flip the majority to class 0 to ensure the result isn't a hardcoded constant.
    let agents2 = vec![
        constant_agent(0, 0),
        constant_agent(1, 0),
        constant_agent(2, 2),
    ];
    assert_eq!(combiner.predict(&agents2, &any_sample()), 0);
}

/// Tie-breaking: 1 vote each for 0, 1, 2 → must return lowest index (0).
/// Also test 1-vs-1 ties at (1,2) and (0,2) for full coverage.
#[test]
fn test_majority_vote_tie_breaking_lowest_index() {
    let v = VotingCombiner::new(VotingStrategy::Majority);

    // 1-1-1 tie
    let agents = vec![
        constant_agent(0, 0),
        constant_agent(1, 1),
        constant_agent(2, 2),
    ];
    assert_eq!(v.predict(&agents, &any_sample()), 0);

    // 1-1 tie between classes 1 and 2 (no class-0 votes) → must return 1
    let agents_12 = vec![constant_agent(0, 1), constant_agent(1, 2)];
    assert_eq!(v.predict(&agents_12, &any_sample()), 1);

    // 1-1 tie between classes 0 and 2 → must return 0
    let agents_02 = vec![constant_agent(0, 0), constant_agent(1, 2)];
    assert_eq!(v.predict(&agents_02, &any_sample()), 0);
}

/// Determinism: same agents + same sample must produce identical predictions
/// across repeated calls (no hidden state, no RNG).
#[test]
fn test_majority_vote_deterministic() {
    let agents = make_agents(7);
    let combiner = VotingCombiner::new(VotingStrategy::Majority);
    let sample = make_sample(vec![0.5, 0.5], 1);
    let first = combiner.predict(&agents, &sample);
    for _ in 0..10 {
        assert_eq!(combiner.predict(&agents, &sample), first);
    }
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

/// Real correctness check for weighted voting: with weights [0.9, 0.1] and
/// agents predicting different labels, the heavy-weight agent MUST win.
#[test]
fn test_weighted_vote_heavy_weight_wins() {
    let agents = vec![constant_agent(0, 2), constant_agent(1, 0)];
    let combiner = VotingCombiner::weighted(vec![0.9, 0.1]);
    assert_eq!(combiner.predict(&agents, &any_sample()), 2);

    // Flip weights: now class 0 should win.
    let combiner_flip = VotingCombiner::weighted(vec![0.1, 0.9]);
    assert_eq!(combiner_flip.predict(&agents, &any_sample()), 0);
}

/// Weighted-vote tie-breaking: equal weights + distinct predictions = tie →
/// lowest index wins.
#[test]
fn test_weighted_vote_tie_breaking_lowest_index() {
    let agents = vec![constant_agent(0, 1), constant_agent(1, 2)];
    let combiner = VotingCombiner::weighted(vec![0.5, 0.5]);
    assert_eq!(combiner.predict(&agents, &any_sample()), 1);
}

/// Verifies the documented fallback: a weighted combiner whose `agent_weights`
/// is shorter than the agent list silently uses each missing agent's `accuracy`.
/// Here both agents have accuracy 0.0 (set by `constant_agent`), so weighted
/// scores are 0.0 for both predicted classes and the lowest-index tie-break
/// decides.
#[test]
fn test_weighted_vote_short_weights_falls_back_to_accuracy() {
    let agents = vec![constant_agent(0, 2), constant_agent(1, 1)];
    // No weights supplied — both agents fall back to accuracy = 0.0.
    let combiner = VotingCombiner::weighted(vec![]);
    // scores = [0.0, 0.0, 0.0]; tie → lowest index, which is 0.
    assert_eq!(combiner.predict(&agents, &any_sample()), 0);
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

/// Hand-derived worked example for BoostingCombiner weight updates.
///
/// Setup:
///   - 2 constant agents: agent 0 always predicts 0, agent 1 always predicts 1.
///   - 3 samples with labels [0, 1, 1].
///   - 1 boosting round, learning_rate = 0.5.
///
/// Hand derivation (matches BoostingCombiner::fit step by step):
///   initial agent_weights  = [0.5, 0.5]
///   initial sample_weights = [1/3, 1/3, 1/3]
///
///   Round 0:
///     agent_errors[0] = sw[1] + sw[2] = 2/3   (agent 0 wrong on samples 1,2)
///     agent_errors[1] = sw[0]         = 1/3   (agent 1 wrong on sample  0)
///     alpha_0 = 0.5 * ln((1 - 2/3)/(2/3)) * 0.5 = 0.25 * ln(0.5) ≈ -0.1733
///     alpha_1 = 0.5 * ln((1 - 1/3)/(1/3)) * 0.5 = 0.25 * ln(2.0) ≈ +0.1733
///     aw[0] += alpha_0  =>  0.5 - 0.1733 = 0.3267
///     aw[1] += alpha_1  =>  0.5 + 0.1733 = 0.6733
///     normalize: sum = 1.0, unchanged
///
///     ensemble prediction for every sample = argmax(scores) where
///       scores = [aw[0], aw[1], 0] = [0.3267, 0.6733, 0]  =>  predicts 1
///     sample 0 (label 0, pred 1, WRONG):  sw[0] *= 1.5      =>  0.5
///     sample 1 (label 1, pred 1, RIGHT):  sw[1] *= 0.75     =>  0.25
///     sample 2 (label 1, pred 1, RIGHT):  sw[2] *= 0.75     =>  0.25
///     normalize: sum = 1.0, unchanged
///
/// Final agent_weights = [0.3267..., 0.6733...]; predict() = 1.
#[test]
fn test_boosting_weight_updates_hand_derived() {
    let agents = vec![constant_agent(0, 0), constant_agent(1, 1)];
    let samples = vec![
        make_sample(vec![0.0, 0.0], 0),
        make_sample(vec![0.0, 0.0], 1),
        make_sample(vec![0.0, 0.0], 1),
    ];
    let mut booster = BoostingCombiner::new(1, 0.5);
    booster.fit(&agents, &samples);

    let aw0_expected = 0.5 + 0.25 * (0.5_f64).ln();
    let aw1_expected = 0.5 + 0.25 * (2.0_f64).ln();
    assert!(
        (booster.agent_weights[0] - aw0_expected).abs() < 1e-12,
        "agent_weights[0] = {}, expected {}",
        booster.agent_weights[0],
        aw0_expected
    );
    assert!(
        (booster.agent_weights[1] - aw1_expected).abs() < 1e-12,
        "agent_weights[1] = {}, expected {}",
        booster.agent_weights[1],
        aw1_expected
    );
    // After this round the ensemble weighted vote for any input must be 1
    // (aw[1] > aw[0], agents constant).
    let probe = make_sample(vec![0.0, 0.0], 0);
    assert_eq!(booster.predict(&agents, &probe), 1);
}

/// After fit, predict on an agent list longer than the trained one must use
/// the documented fallback weight `1.0 / agents.len()` for the unseen agents
/// (here: agent 2 has no learned weight, so its vote counts as 1/3).
#[test]
fn test_boosting_predict_fallback_weight_for_unfitted_agents() {
    let trained = vec![constant_agent(0, 0), constant_agent(1, 1)];
    let samples = vec![make_sample(vec![0.0, 0.0], 0)];
    let mut booster = BoostingCombiner::new(1, 0.5);
    booster.fit(&trained, &samples);
    // Trained on 2 agents; predict on 3 — the third uses fallback weight 1/3.
    let probe_agents = vec![
        constant_agent(0, 0),
        constant_agent(1, 1),
        constant_agent(2, 2),
    ];
    let pred = booster.predict(&probe_agents, &make_sample(vec![0.0, 0.0], 0));
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

/// Hand-computed accuracy: a single constant-0 agent classifies every sample
/// as 0. Out of 3 samples (one per true label), exactly one is correct.
/// accuracy = 1/3, ensemble_correct = 1, n_samples = 3.
#[test]
fn test_evaluator_accuracy_hand_computed() {
    let agents = vec![constant_agent(0, 0)];
    let ensemble = Ensemble::new(
        agents,
        CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority)),
    );
    let samples = vec![
        make_sample(vec![0.0, 0.0], 0),
        make_sample(vec![0.0, 0.0], 1),
        make_sample(vec![0.0, 0.0], 2),
    ];
    let result = ensemble.evaluator().evaluate(&samples);
    assert_eq!(result.n_samples, 3);
    assert_eq!(result.ensemble_correct, 1);
    assert!((result.ensemble_accuracy - 1.0 / 3.0).abs() < 1e-12);
    // Conservation check: correct + wrong == n_samples
    assert_eq!(
        result.ensemble_correct + result.n_samples - result.ensemble_correct,
        result.n_samples
    );
}

/// Empty dataset: documented behavior is to return 0.0 accuracy and a
/// zero-filled confusion matrix rather than panic.
#[test]
fn test_evaluator_empty_dataset() {
    let agents = vec![constant_agent(0, 0)];
    let ensemble = Ensemble::new(
        agents,
        CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority)),
    );
    assert_eq!(ensemble.evaluator().accuracy(&[]), 0.0);
    let result = ensemble.evaluator().evaluate(&[]);
    assert_eq!(result.n_samples, 0);
    assert_eq!(result.ensemble_correct, 0);
    assert!(result.ensemble_accuracy.is_nan() || result.ensemble_accuracy == 0.0);
    for r in 0..3 {
        for c in 0..3 {
            assert_eq!(result.confusion_matrix[r][c], 0);
        }
    }
}

/// Hand-computed confusion matrix:
///   - All 3 samples are predicted as 0 (single constant-0 agent).
///   - True labels are 0, 1, 2 (one each).
///   - confusion[p][a]: predicted=0, actual=0,1,2 each appear once.
///     => confusion[0] = [1, 1, 1], all other rows zero.
#[test]
fn test_evaluator_confusion_matrix_hand_computed() {
    let agents = vec![constant_agent(0, 0)];
    let ensemble = Ensemble::new(
        agents,
        CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority)),
    );
    let samples = vec![
        make_sample(vec![0.0, 0.0], 0),
        make_sample(vec![0.0, 0.0], 1),
        make_sample(vec![0.0, 0.0], 2),
    ];
    let result = ensemble.evaluator().evaluate(&samples);
    assert_eq!(result.confusion_matrix[0], [1, 1, 1]);
    assert_eq!(result.confusion_matrix[1], [0, 0, 0]);
    assert_eq!(result.confusion_matrix[2], [0, 0, 0]);
}

/// Hand-computed precision/recall on a known confusion matrix.
///
/// Constructed matrix [predicted][actual]:
///   ```text
///           actual 0  actual 1  actual 2
///   pred 0:    2         1         0       -> precision[0] = 2/3
///   pred 1:    0         3         1       -> precision[1] = 3/4
///   pred 2:    0         0         1       -> precision[2] = 1/1 = 1.0
///
///   actual totals: 2, 4, 2
///   recall[0] = 2/2 = 1.0
///   recall[1] = 3/4 = 0.75
///   recall[2] = 1/2 = 0.5
///   ```
#[test]
fn test_evaluator_precision_recall_hand_computed() {
    let result = EvaluationResult {
        ensemble_accuracy: 0.0,
        individual_accuracies: vec![],
        n_samples: 7,
        ensemble_correct: 6,
        confusion_matrix: [[2, 1, 0], [0, 3, 1], [0, 0, 1]],
    };
    let precision = result.precision();
    let recall = result.recall();

    assert!((precision[0] - 2.0 / 3.0).abs() < 1e-12);
    assert!((precision[1] - 3.0 / 4.0).abs() < 1e-12);
    assert!((precision[2] - 1.0).abs() < 1e-12);

    assert!((recall[0] - 1.0).abs() < 1e-12);
    assert!((recall[1] - 0.75).abs() < 1e-12);
    assert!((recall[2] - 0.5).abs() < 1e-12);
}

/// Precision/recall on a confusion matrix with an entirely-empty row and
/// column: class 1 never appears, so precision[1] and recall[1] must remain 0.0
/// (the documented divide-by-zero guard) rather than NaN.
#[test]
fn test_evaluator_precision_recall_unseen_class_is_zero_not_nan() {
    let result = EvaluationResult {
        ensemble_accuracy: 0.0,
        individual_accuracies: vec![],
        n_samples: 2,
        ensemble_correct: 2,
        confusion_matrix: [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
    };
    let precision = result.precision();
    let recall = result.recall();
    assert!((precision[0] - 1.0).abs() < 1e-12);
    assert!(!precision[1].is_nan());
    assert_eq!(precision[1], 0.0);
    assert!((precision[2] - 1.0).abs() < 1e-12);
    assert!((recall[0] - 1.0).abs() < 1e-12);
    assert_eq!(recall[1], 0.0);
    assert!((recall[2] - 1.0).abs() < 1e-12);
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
    // Sanity: improvement_over_best must equal (ensemble_acc - max(individual)).
    // Verify the invariant by hand: ensemble_acc - max(agent_acc).
    let best = result
        .individual_accuracies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((result.improvement_over_best() - (result.ensemble_accuracy - best)).abs() < 1e-12);
    // improvement_over_average: ensemble_acc - mean(individual).
    let avg = result.individual_accuracies.iter().sum::<f64>()
        / result.individual_accuracies.len() as f64;
    assert!((result.improvement_over_average() - (result.ensemble_accuracy - avg)).abs() < 1e-12);
}

/// improvement_over_average on an empty individual_accuracies vector must
/// short-circuit to 0.0 (documented) rather than divide by zero.
#[test]
fn test_evaluator_improvement_over_average_empty() {
    let result = EvaluationResult {
        ensemble_accuracy: 0.5,
        individual_accuracies: vec![],
        n_samples: 0,
        ensemble_correct: 0,
        confusion_matrix: [[0; 3]; 3],
    };
    assert_eq!(result.improvement_over_average(), 0.0);
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
        assert!((0.0..=1.0).contains(&precision[c]));
        assert!((0.0..=1.0).contains(&recall[c]));
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
