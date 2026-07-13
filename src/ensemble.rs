//! Ensemble container that holds weak agents and delegates to a combination strategy.

use crate::{
    BoostingCombiner, EnsembleEvaluator, StackingCombiner, TernaryLabel, TernarySample,
    VotingCombiner,
};

/// Strategy for combining weak agent predictions.
#[derive(Debug, Clone)]
pub enum CombineStrategy {
    /// Majority or weighted voting.
    Voting(VotingCombiner),
    /// Sequential boosting.
    Boosting(BoostingCombiner),
    /// Meta-learner stacking.
    Stacking(StackingCombiner),
}

/// An ensemble of weak ternary agents combined with a strategy.
#[derive(Debug, Clone)]
pub struct Ensemble {
    /// The weak agents in this ensemble.
    pub agents: Vec<crate::WeakAgent>,
    /// The combination strategy.
    pub strategy: CombineStrategy,
}

impl Ensemble {
    /// Create a new ensemble with the given agents and strategy.
    pub fn new(agents: Vec<crate::WeakAgent>, strategy: CombineStrategy) -> Self {
        assert!(!agents.is_empty(), "Ensemble must have at least one agent");
        Self { agents, strategy }
    }

    /// Predict using the ensemble's combination strategy.
    pub fn predict(&self, sample: &TernarySample) -> TernaryLabel {
        match &self.strategy {
            CombineStrategy::Voting(v) => v.predict(&self.agents, sample),
            CombineStrategy::Boosting(b) => b.predict(&self.agents, sample),
            CombineStrategy::Stacking(s) => s.predict(&self.agents, sample),
        }
    }

    /// Predict for multiple samples.
    pub fn predict_batch(&self, samples: &[TernarySample]) -> Vec<TernaryLabel> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Get the number of agents in this ensemble.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if the ensemble is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Create an evaluator for this ensemble.
    pub fn evaluator(&self) -> EnsembleEvaluator {
        EnsembleEvaluator::new(self.clone())
    }
}
