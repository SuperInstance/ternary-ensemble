//! Voting-based combination strategies.
//!
//! # Tie-breaking convention
//!
//! All three strategies resolve ties toward the **lowest class index**: if two
//! classes receive equal support, class 0 beats class 1 beats class 2. This is
//! enforced by iterating the per-class scores in reverse and using
//! `Iterator::max_by` / `min_by` (which return the *last* element on ties), so
//! the lowest index — visited last — wins. Comparisons use `f64::total_cmp`
//! everywhere, which gives a fully-defined order even when NaN scores
//! accidentally arise (NaN sorts greater than every finite value under
//! `total_cmp`, so a NaN score would lose to any finite score under
//! max-selection and win under min-selection — predictable, not silent).

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
    /// Optional per-agent weights (used with the [`Weighted`](VotingStrategy::Weighted) strategy).
    ///
    /// If the vector is shorter than the agent list passed to [`predict`](Self::predict),
    /// the missing weights fall back to each agent's `accuracy` field. This
    /// silent fallback is intentional but worth noting: if you intended to
    /// weight every agent explicitly, pass a vector of the same length as the
    /// agent slice.
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
        // Reverse-iterate so max_by_key's "last wins" tie rule resolves to the
        // lowest class index. See the crate-level tie-breaking note.
        counts
            .iter()
            .enumerate()
            .rev()
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
        // total_cmp() for fully-defined ordering; reverse-iterate for
        // lowest-index-wins tie-breaking (see crate-level note).
        scores
            .iter()
            .enumerate()
            .rev()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
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
            confidences.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (rank, (label, _)) in confidences.iter().enumerate() {
                borda_scores[*label as usize] += rank as f64;
            }
        }

        // Lowest Borda score wins. Reverse-iterate so ties resolve to the
        // lowest class index (min_by returns the last min element).
        borda_scores
            .iter()
            .enumerate()
            .rev()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as TernaryLabel)
            .unwrap_or(0)
    }
}
