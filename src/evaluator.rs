//! Ensemble evaluator — compare ensemble performance against individual agents.

use crate::{Ensemble, TernarySample};

/// Result of evaluating an ensemble.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Accuracy of the ensemble (0.0 to 1.0).
    pub ensemble_accuracy: f64,
    /// Accuracy of each individual agent.
    pub individual_accuracies: Vec<f64>,
    /// Number of samples evaluated.
    pub n_samples: usize,
    /// Number of correct ensemble predictions.
    pub ensemble_correct: usize,
    /// Confusion matrix for the ensemble: `confusion_matrix[predicted][actual]`.
    pub confusion_matrix: [[usize; 3]; 3],
}

impl EvaluationResult {
    /// How much the ensemble improves over the best individual agent.
    pub fn improvement_over_best(&self) -> f64 {
        let best_individual = self
            .individual_accuracies
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        self.ensemble_accuracy - best_individual
    }

    /// How much the ensemble improves over the average individual agent.
    pub fn improvement_over_average(&self) -> f64 {
        if self.individual_accuracies.is_empty() {
            return 0.0;
        }
        let avg: f64 = self.individual_accuracies.iter().sum::<f64>()
            / self.individual_accuracies.len() as f64;
        self.ensemble_accuracy - avg
    }

    /// Per-class precision: of samples predicted as class c, how many were correct?
    pub fn precision(&self) -> [f64; 3] {
        let mut result = [0.0f64; 3];
        for (c, result_c) in result.iter_mut().enumerate() {
            let total_predicted: usize = (0..3).map(|r| self.confusion_matrix[c][r]).sum();
            if total_predicted > 0 {
                *result_c = self.confusion_matrix[c][c] as f64 / total_predicted as f64;
            }
        }
        result
    }

    /// Per-class recall: of samples truly in class c, how many were predicted correctly?
    pub fn recall(&self) -> [f64; 3] {
        let mut result = [0.0f64; 3];
        for (c, result_c) in result.iter_mut().enumerate() {
            let total_actual: usize = (0..3).map(|p| self.confusion_matrix[p][c]).sum();
            if total_actual > 0 {
                *result_c = self.confusion_matrix[c][c] as f64 / total_actual as f64;
            }
        }
        result
    }
}

/// Evaluator for comparing ensemble vs individual agent performance.
#[derive(Debug, Clone)]
pub struct EnsembleEvaluator {
    /// The ensemble to evaluate.
    pub ensemble: Ensemble,
}

impl EnsembleEvaluator {
    /// Create a new evaluator for the given ensemble.
    pub fn new(ensemble: Ensemble) -> Self {
        Self { ensemble }
    }

    /// Evaluate the ensemble on the given test samples.
    pub fn evaluate(&self, samples: &[TernarySample]) -> EvaluationResult {
        let n_samples = samples.len();
        let mut ensemble_correct = 0usize;
        let mut confusion = [[0usize; 3]; 3];
        let mut individual_correct = vec![0usize; self.ensemble.agents.len()];

        for sample in samples {
            // Ensemble prediction
            let pred = self.ensemble.predict(sample);
            confusion[pred as usize][sample.label as usize] += 1;
            if pred == sample.label {
                ensemble_correct += 1;
            }

            // Individual predictions
            for (i, agent) in self.ensemble.agents.iter().enumerate() {
                if agent.predict(sample) == sample.label {
                    individual_correct[i] += 1;
                }
            }
        }

        let ensemble_accuracy = if n_samples > 0 {
            ensemble_correct as f64 / n_samples as f64
        } else {
            0.0
        };

        let individual_accuracies: Vec<f64> = individual_correct
            .iter()
            .map(|&c| {
                if n_samples > 0 {
                    c as f64 / n_samples as f64
                } else {
                    0.0
                }
            })
            .collect();

        EvaluationResult {
            ensemble_accuracy,
            individual_accuracies,
            n_samples,
            ensemble_correct,
            confusion_matrix: confusion,
        }
    }

    /// Quick accuracy check on a small set.
    pub fn accuracy(&self, samples: &[TernarySample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let correct = samples
            .iter()
            .filter(|s| self.ensemble.predict(s) == s.label)
            .count();
        correct as f64 / samples.len() as f64
    }

    /// Compare ensemble accuracy vs each individual agent.
    pub fn compare(&self, samples: &[TernarySample]) -> Vec<(usize, f64, f64)> {
        let ens_acc = self.accuracy(samples);
        self.ensemble
            .agents
            .iter()
            .enumerate()
            .map(|(i, agent)| {
                let agent_correct = samples
                    .iter()
                    .filter(|s| agent.predict(s) == s.label)
                    .count();
                let agent_acc = if samples.is_empty() {
                    0.0
                } else {
                    agent_correct as f64 / samples.len() as f64
                };
                (i, agent_acc, ens_acc)
            })
            .collect()
    }
}
