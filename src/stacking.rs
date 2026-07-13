//! Stacking combiner — meta-learner that combines base agent outputs.

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
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0);

        (logits, predicted)
    }

    /// Predict using the meta-learner.
    pub fn predict(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        if !self.fitted {
            // Fallback to simple majority vote if not fitted
            let mut counts = [0usize; 3];
            for agent in agents {
                let pred = agent.predict(sample);
                counts[pred as usize] += 1;
            }
            return counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i as TernaryLabel)
                .unwrap_or(0);
        }

        let meta_features: Vec<[f64; 3]> = vec![[0.0; 3]; agents.len()];
        let (_, predicted) = self.compute_meta_logits(agents, sample, &meta_features);
        predicted
    }
}
