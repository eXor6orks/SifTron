//! The `IndexStrategy` interface every index backend (`Flat`, and later
//! `Ivf`, `Hnsw`) implements. Per the project's additive-not-regressive
//! principle, backends are selectable per collection rather than replacing
//! one another.

use crate::record::{Hit, Record};

pub trait IndexStrategy {
    /// Inserts or overwrites the record with the given `id`.
    fn upsert(&mut self, record: Record);

    /// Removes the record with the given `id`, if present. Returns whether
    /// a record was actually removed.
    fn remove(&mut self, id: u64) -> bool;

    /// Returns the `k` nearest records to `query`, sorted by ascending
    /// distance (closest first). May return fewer than `k` results if the
    /// index holds fewer than `k` records.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;

    /// Number of records currently held by the index.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
