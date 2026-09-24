//! Data types shared by every `IndexStrategy` implementation.

use std::collections::HashMap;

/// Metadata attached to a [`Record`]: named fields (e.g. `categorie`,
/// `prix`, `en_stock`), mirroring the `payload_schema` described in the
/// project plan for combined filtering (v0.6.0). Filtering on these fields
/// isn't implemented yet — for now they just travel with the record.
pub type Payload = HashMap<String, serde_json::Value>;

/// A vector stored in a collection, identified by a stable `u64` id, with
/// an optional payload of metadata fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Payload,
}

impl Record {
    /// Creates a record with no metadata.
    pub fn new(id: u64, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            payload: Payload::new(),
        }
    }

    /// Attaches a payload, replacing any previously set metadata.
    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = payload;
        self
    }
}

/// A single search result: the id of a matching record and its distance to
/// the query vector (lower is closer, per the [`crate::Distance`] contract).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_record_has_empty_payload() {
        let record = Record::new(1, vec![1.0, 2.0]);
        assert!(record.payload.is_empty());
    }

    #[test]
    fn with_payload_attaches_metadata() {
        let mut payload = Payload::new();
        payload.insert("categorie".into(), serde_json::json!("chaussures"));
        payload.insert("prix".into(), serde_json::json!(49.99));

        let record = Record::new(1, vec![1.0, 2.0]).with_payload(payload);

        assert_eq!(
            record.payload.get("categorie"),
            Some(&serde_json::json!("chaussures"))
        );
    }
}
