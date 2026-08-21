//! Boosting combiner — sequential reweighting of misclassified samples.

use crate::{TernaryLabel, TernarySample, WeakAgent};

/// A combiner that implements a boosting approach (simplified AdaBoost for ternary).
///
/// Agents are weighted based on their accuracy, and misclassified samples
/// get higher weight in subsequent rounds.
#[derive(Debug, Clone)]
pub struct BoostingCombiner {
    /// Weight for each agent (learned during fitting).
    pub agent_weights: Vec<f64>,
    /// Number of boosting rounds.
    pub rounds: usize,
    /// Learning rate for weight updates.
    pub learning_rate: f64,
}

impl BoostingCombiner {
    /// Create a new boosting combiner with uniform initial weights.
    pub fn new(rounds: usize, learning_rate: f64) -> Self {
        Self {
            agent_weights: Vec::new(),
            rounds,
            learning_rate,
        }
    }

    /// Fit the booster on training data to learn agent weights.
    ///
    /// # Panics
    /// Panics if `agents` or `samples` is empty. Boosting on an empty set is
    /// degenerate (initial weights would involve division by zero, and no
    /// error signal could ever be produced), so we treat it as a programming
    /// error rather than silently producing a meaningless model.
    pub fn fit(&mut self, agents: &[WeakAgent], samples: &[TernarySample]) {
        assert!(
            !agents.is_empty(),
            "BoostingCombiner::fit requires at least one agent"
        );
        assert!(
            !samples.is_empty(),
            "BoostingCombiner::fit requires at least one sample"
        );
        let n_agents = agents.len();
        let n_samples = samples.len();

        // Initialize agent weights uniformly
        self.agent_weights = vec![1.0 / n_agents as f64; n_agents];

        // Sample weights start uniform
        let mut sample_weights = vec![1.0 / n_samples as f64; n_samples];

        for _round in 0..self.rounds {
            // Evaluate each agent on weighted samples
            let mut agent_errors = vec![0.0f64; n_agents];

            for (s_idx, sample) in samples.iter().enumerate() {
                for (a_idx, agent) in agents.iter().enumerate() {
                    let pred = agent.predict(sample);
                    if pred != sample.label {
                        agent_errors[a_idx] += sample_weights[s_idx];
                    }
                }
            }

            // Update agent weights based on error
            for (a_idx, error) in agent_errors.iter().enumerate() {
                let err = error.clamp(1e-10, 1.0 - 1e-10);
                let alpha = 0.5 * ((1.0 - err) / err).ln() * self.learning_rate;
                self.agent_weights[a_idx] += alpha;
            }

            // Normalize agent weights
            let sum: f64 = self.agent_weights.iter().sum();
            if sum > 0.0 {
                for w in &mut self.agent_weights {
                    *w /= sum;
                }
            }

            // Update sample weights: increase for misclassified, decrease for correct
            for (s_idx, sample) in samples.iter().enumerate() {
                // Weighted ensemble prediction
                let mut scores = [0.0f64; 3];
                for (a_idx, agent) in agents.iter().enumerate() {
                    let pred = agent.predict(sample);
                    scores[pred as usize] += self.agent_weights[a_idx];
                }
                let ensemble_pred = scores
                    .iter()
                    .enumerate()
                    .rev()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as TernaryLabel)
                    .unwrap_or(0);

                if ensemble_pred != sample.label {
                    sample_weights[s_idx] *= 1.0 + self.learning_rate;
                } else {
                    sample_weights[s_idx] *= 1.0 - self.learning_rate * 0.5;
                }
            }

            // Normalize sample weights
            let sw_sum: f64 = sample_weights.iter().sum();
            if sw_sum > 0.0 {
                for w in &mut sample_weights {
                    *w /= sw_sum;
                }
            }
        }
    }

    /// Predict using the boosted weighted vote.
    pub fn predict(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        let mut scores = [0.0f64; 3];

        for (i, agent) in agents.iter().enumerate() {
            let pred = agent.predict(sample);
            let w = if i < self.agent_weights.len() {
                self.agent_weights[i]
            } else {
                1.0 / agents.len() as f64
            };
            scores[pred as usize] += w;
        }

        scores
            .iter()
            .enumerate()
            .rev()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0)
    }
}
