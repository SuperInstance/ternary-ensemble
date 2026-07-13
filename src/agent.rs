//! Weak agent with known accuracy < 60%.

use crate::{TernaryLabel, TernarySample};

/// A simple weak agent that predicts ternary labels.
///
/// Each agent has a known accuracy (below 60%) and a deterministic
/// prediction function based on a linear threshold of input features.
#[derive(Debug, Clone)]
pub struct WeakAgent {
    /// Unique identifier for this agent.
    pub id: usize,
    /// Weight coefficients for each feature dimension.
    pub weights: Vec<f64>,
    /// Bias term.
    bias: f64,
    /// Known accuracy of this agent (0.0 to 1.0, always < 0.60).
    pub accuracy: f64,
    /// Optional noise seed reserved for deterministic-but-noisy predictions.
    ///
    /// Currently unread; retained on the struct so future perturbation strategies
    /// can be added without changing the public constructor signature.
    #[allow(dead_code)]
    noise_seed: u64,
}

impl WeakAgent {
    /// Create a new weak agent with the given weights, bias, and accuracy.
    ///
    /// # Panics
    /// Panics if accuracy >= 0.60 or accuracy < 0.0.
    pub fn new(id: usize, weights: Vec<f64>, bias: f64, accuracy: f64, noise_seed: u64) -> Self {
        assert!(
            (0.0..0.60).contains(&accuracy),
            "Weak agent accuracy must be in [0.0, 0.60), got {}",
            accuracy
        );
        Self {
            id,
            weights,
            bias,
            accuracy,
            noise_seed,
        }
    }

    /// Create a weak agent with random-ish weights and specified accuracy.
    pub fn with_accuracy(id: usize, feature_dim: usize, accuracy: f64, seed: u64) -> Self {
        let mut s = seed;
        let weights: Vec<f64> = (0..feature_dim)
            .map(|i| {
                s = s.wrapping_add(i as u64).wrapping_mul(6364136223846793005);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect();
        let bias = (seed as f64 % 1.0) - 0.5;
        Self::new(id, weights, bias, accuracy, seed)
    }

    /// Predict the label for a single sample.
    pub fn predict(&self, sample: &TernarySample) -> TernaryLabel {
        let score = self.compute_score(&sample.features);
        // Simple threshold-based ternary classification
        if score < -0.3 {
            0
        } else if score < 0.3 {
            1
        } else {
            2
        }
    }

    /// Predict labels for multiple samples.
    pub fn predict_batch(&self, samples: &[TernarySample]) -> Vec<TernaryLabel> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Get the raw score for a feature vector.
    fn compute_score(&self, features: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum::<f64>()
            + self.bias
    }

    /// Get a confidence-weighted prediction (label, confidence).
    pub fn predict_with_confidence(&self, sample: &TernarySample) -> (TernaryLabel, f64) {
        let score = self.compute_score(&sample.features);
        let label = if score < -0.3 {
            0
        } else if score < 0.3 {
            1
        } else {
            2
        };
        // Confidence based on distance from threshold
        let confidence = (score.abs() * self.accuracy).min(1.0);
        (label, confidence)
    }
}
