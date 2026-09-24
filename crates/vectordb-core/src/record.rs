//! Data types shared by every `IndexStrategy` implementation.

/// A vector stored in a collection, identified by a stable `u64` id.
///
/// Payload fields (for combined filtering, see the v0.6.0 milestone) are
/// intentionally absent here — this type covers only what the vector-only
/// baseline (v0.1.0) needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub id: u64,
    pub vector: Vec<f32>,
}

impl Record {
    pub fn new(id: u64, vector: Vec<f32>) -> Self {
        Self { id, vector }
    }
}

/// A single search result: the id of a matching record and its distance to
/// the query vector (lower is closer, per the [`crate::Distance`] contract).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub score: f32,
}
