//! # ternary-ensemble
//!
//! Ensemble methods for ternary agents — combine multiple weak agents into a strong one.
//!
//! Ternary agents produce one of three possible outputs (represented as `0, 1, 2`).
//! This crate provides ensemble strategies (voting, boosting, stacking) to combine
//! multiple weak agents into a single stronger predictor.
//!
//! # Example
//!
//! ```
//! use ternary_ensemble::{
//!     CombineStrategy, Ensemble, TernarySample, VotingCombiner, VotingStrategy, WeakAgent,
//! };
//!
//! let agents: Vec<WeakAgent> = (0..5)
//!     .map(|i| WeakAgent::with_accuracy(i, 2, 0.40, i as u64 * 17 + 3))
//!     .collect();
//! let strategy =
//!     CombineStrategy::Voting(VotingCombiner::new(VotingStrategy::Majority));
//! let ensemble = Ensemble::new(agents, strategy);
//!
//! let probe = TernarySample::new(vec![0.1, 0.2], 0);
//! let prediction = ensemble.predict(&probe);
//! assert!(prediction <= 2);
//! ```

mod agent;
mod boosting;
mod ensemble;
mod evaluator;
mod stacking;
mod voting;

pub use agent::WeakAgent;
pub use boosting::BoostingCombiner;
pub use ensemble::{CombineStrategy, Ensemble};
pub use evaluator::{EnsembleEvaluator, EvaluationResult};
pub use stacking::StackingCombiner;
pub use voting::{VotingCombiner, VotingStrategy};

/// A ternary label: one of three classes.
pub type TernaryLabel = u8;

/// A sample index used in training sets.
pub type SampleIndex = usize;

/// A ternary sample with features and a label.
#[derive(Debug, Clone)]
pub struct TernarySample {
    /// Feature vector (simple f64 values).
    pub features: Vec<f64>,
    /// True label in {0, 1, 2}.
    pub label: TernaryLabel,
}

impl TernarySample {
    /// Create a new ternary sample.
    pub fn new(features: Vec<f64>, label: TernaryLabel) -> Self {
        assert!(label <= 2, "Ternary label must be 0, 1, or 2");
        Self { features, label }
    }
}
