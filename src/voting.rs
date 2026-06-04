//! Voting-based combination strategies.

use crate::{TernaryLabel, TernarySample, WeakAgent};

/// The type of voting to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VotingStrategy {
    /// Simple majority vote — each agent gets one equal vote.
    Majority,
    /// Weighted vote — each agent's vote is weighted by its accuracy.
    Weighted,
    /// Ranked vote — agents rank classes; lowest average rank wins (Borda count).
    Ranked,
}

/// A combiner that uses voting to aggregate weak agent predictions.
#[derive(Debug, Clone)]
pub struct VotingCombiner {
    /// The voting strategy to use.
    pub strategy: VotingStrategy,
    /// Optional per-agent weights (used with Weighted strategy).
    pub agent_weights: Vec<f64>,
}

impl VotingCombiner {
    /// Create a new voting combiner with the specified strategy.
    pub fn new(strategy: VotingStrategy) -> Self {
        Self {
            strategy,
            agent_weights: Vec::new(),
        }
    }

    /// Create a weighted voting combiner with explicit agent weights.
    pub fn weighted(agent_weights: Vec<f64>) -> Self {
        Self {
            strategy: VotingStrategy::Weighted,
            agent_weights,
        }
    }

    /// Predict using voting.
    pub fn predict(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        match self.strategy {
            VotingStrategy::Majority => self.majority_vote(agents, sample),
            VotingStrategy::Weighted => self.weighted_vote(agents, sample),
            VotingStrategy::Ranked => self.ranked_vote(agents, sample),
        }
    }

    fn majority_vote(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        let mut counts = [0usize; 3];
        for agent in agents {
            let pred = agent.predict(sample);
            counts[pred as usize] += 1;
        }
        counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0)
    }

    fn weighted_vote(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        let mut scores = [0.0f64; 3];
        for (i, agent) in agents.iter().enumerate() {
            let pred = agent.predict(sample);
            let w = if i < self.agent_weights.len() {
                self.agent_weights[i]
            } else {
                agent.accuracy
            };
            scores[pred as usize] += w;
        }
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0)
    }

    fn ranked_vote(&self, agents: &[WeakAgent], sample: &TernarySample) -> TernaryLabel {
        // Borda count: each agent assigns rank 0 (best) to 2 (worst) for each class.
        // Rank is determined by confidence-based ordering.
        let mut borda_scores = [0.0f64; 3];

        for agent in agents {
            let mut confidences: Vec<(TernaryLabel, f64)> = Vec::with_capacity(3);
            // Compute a simple confidence per class based on the agent's score
            for label in 0u8..=2 {
                let fake = TernarySample::new(sample.features.clone(), label);
                let (_, conf) = agent.predict_with_confidence(&fake);
                confidences.push((label, conf));
            }
            // Sort by confidence descending; highest confidence gets rank 0
            confidences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (rank, (label, _)) in confidences.iter().enumerate() {
                borda_scores[*label as usize] += rank as f64;
            }
        }

        // Lowest Borda score wins
        borda_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0)
    }
}
