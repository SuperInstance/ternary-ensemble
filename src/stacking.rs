//! Stacking combiner — meta-learner that combines base agent outputs.
//!
//! # On data leakage
//!
//! Classic stacking leakage happens when base models are trained on dataset D
//! and then asked to produce the meta-features used to train the meta-learner
//! on that same D — the base predictions are overfit to D, so the meta-learner
//! learns weights that won't generalize. The standard fix is to feed the
//! meta-learner only out-of-fold base predictions (or predictions on a held-out
//! split).
//!
//! That risk **does not apply** to this combiner, because [`WeakAgent`] has no
//! `fit` method: agents are constructed with fixed `weights`, `bias`, and
//! `accuracy` via [`WeakAgent::new`] / [`WeakAgent::with_accuracy`]. A base
//! agent's `predict(sample)` is therefore identical whether `sample` came from
//! the stacking fit set or from a held-out test set — there is no training-time
//! overfit on the fit samples to leak. The only thing being learned here is the
//! meta-learner's weight matrix, which is the same statistical overfitting risk
//! any gradient-descent-trained model has, not a stacking-specific leak.
//!
//! If future changes add a trainable base model (so that `WeakAgent::fit`
//! exists and is invoked by the stacking combiner), this reasoning no longer
//! holds and the combiner must be updated to use out-of-fold predictions.

use crate::{TernaryLabel, TernarySample, WeakAgent};

/// A simple meta-learner that learns weights for combining base agent outputs.
#[derive(Debug, Clone)]
pub struct StackingCombiner {
    /// Meta-weights: [n_agents x 3] — weight of each agent for each class.
    /// Stored flat: agent i, class j => meta_weights[i * 3 + j]
    pub meta_weights: Vec<f64>,
    /// Meta-bias terms for each class.
    pub meta_bias: Vec<f64>,
    /// Whether the meta-learner has been fitted.
    pub fitted: bool,
    /// Learning rate for meta-learner training.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
}

impl StackingCombiner {
    /// Create a new stacking combiner.
    pub fn new(learning_rate: f64, epochs: usize) -> Self {
        Self {
            meta_weights: Vec::new(),
            meta_bias: vec![0.0; 3],
            fitted: false,
            learning_rate,
            epochs,
        }
    }

    /// Create a stacking combiner with default hyperparameters.
    pub fn default_params() -> Self {
        Self::new(0.01, 100)
    }

    /// Fit the meta-learner on training data.
    ///
    /// # Panics
    /// Panics if `agents` or `samples` is empty. With zero samples the
    /// gradient-averaging step (`grad / samples.len()`) would compute `0.0 / 0`
    /// and silently poison `meta_bias` with NaN, after which every `predict`
    /// call would return 2 regardless of input. We refuse that silently-broken
    /// state up-front.
    pub fn fit(&mut self, agents: &[WeakAgent], samples: &[TernarySample]) {
        assert!(
            !agents.is_empty(),
            "StackingCombiner::fit requires at least one agent"
        );
        assert!(
            !samples.is_empty(),
            "StackingCombiner::fit requires at least one sample"
        );
        let n_agents = agents.len();
        let n_classes = 3usize;

        // Initialize meta-weights
        self.meta_weights = vec![0.01; n_agents * n_classes];
        self.meta_bias = vec![0.0; n_classes];

        // Generate base agent predictions as meta-features (one-hot encoded)
        let meta_features: Vec<[f64; 3]> = agents
            .iter()
            .map(|agent| {
                // For each agent, compute per-class score based on predictions over all samples
                let mut scores = [0.0f64; 3];
                for sample in samples {
                    let pred = agent.predict(sample);
                    scores[pred as usize] += 1.0;
                }
                let total: f64 = scores.iter().sum();
                if total > 0.0 {
                    for s in &mut scores {
                        *s /= total;
                    }
                }
                scores
            })
            .collect();

        // Train with simple gradient descent on cross-entropy-like loss
        for _epoch in 0..self.epochs {
            let mut grad_weights = vec![0.0f64; n_agents * n_classes];
            let mut grad_bias = [0.0f64; 3];

            for sample in samples {
                // Compute meta-learner prediction for this sample
                let (logits, predicted) = self.compute_meta_logits(agents, sample, &meta_features);

                if predicted != sample.label {
                    // Update: increase weight for correct class, decrease for predicted
                    for (a_idx, mf) in meta_features.iter().enumerate() {
                        for c in 0..n_classes {
                            let target = if c == sample.label as usize {
                                1.0
                            } else {
                                -1.0
                            };
                            grad_weights[a_idx * n_classes + c] +=
                                self.learning_rate * (logits[c] - target) * mf[c];
                        }
                    }
                    for c in 0..n_classes {
                        let target = if c == sample.label as usize {
                            1.0
                        } else {
                            -1.0
                        };
                        grad_bias[c] += self.learning_rate * (logits[c] - target);
                    }
                }
            }

            // Apply gradients
            for (i, w) in self.meta_weights.iter_mut().enumerate() {
                *w -= grad_weights[i] / samples.len() as f64;
            }
            for (i, b) in self.meta_bias.iter_mut().enumerate() {
                *b -= grad_bias[i] / samples.len() as f64;
            }
        }

        self.fitted = true;
    }

    /// Compute meta-learner logits for a sample.
    fn compute_meta_logits(
        &self,
        agents: &[WeakAgent],
        sample: &TernarySample,
        _meta_features: &[[f64; 3]],
    ) -> ([f64; 3], TernaryLabel) {
        let mut logits = [0.0f64; 3];

        for (a_idx, agent) in agents.iter().enumerate() {
            let pred = agent.predict(sample);
            for (c, logit_c) in logits.iter_mut().enumerate() {
                let w_idx = a_idx * 3 + c;
                let w = if w_idx < self.meta_weights.len() {
                    self.meta_weights[w_idx]
                } else {
                    0.0
                };
                // One-hot contribution from agent prediction
                *logit_c += w * if c == pred as usize { 1.0 } else { 0.0 };
            }
        }

        for (c, logit_c) in logits.iter_mut().enumerate() {
            *logit_c += self.meta_bias.get(c).copied().unwrap_or(0.0);
        }

        let predicted = logits
            .iter()
            .enumerate()
            .rev()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0);

        (logits, predicted)
    }

    /// Predict using the meta-learner.
    ///
    /// If the combiner has not been fitted yet, this silently falls back to a
    /// simple majority vote over the supplied agents (matching
    /// [`VotingStrategy::Majority`](crate::VotingStrategy::Majority) semantics,
    /// including lowest-class-index tie-breaking). This fallback exists so that
    /// a freshly-constructed `StackingCombiner` can still be used inside an
    /// [`Ensemble`](crate::Ensemble) before `fit` is called, but callers that
    /// care about the meta-learner's contribution should always call `fit`
    /// first — the fallback is clearly labelled via [`fitted`](Self::fitted).
    pub fn predict(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        if !self.fitted {
            // Fallback to simple majority vote if not fitted.
            let mut counts = [0usize; 3];
            for agent in agents {
                let pred = agent.predict(sample);
                counts[pred as usize] += 1;
            }
            return counts
                .iter()
                .enumerate()
                .rev()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i as TernaryLabel)
                .unwrap_or(0);
        }

        let meta_features: Vec<[f64; 3]> = vec![[0.0; 3]; agents.len()];
        let (_, predicted) = self.compute_meta_logits(agents, sample, &meta_features);
        predicted
    }
}
