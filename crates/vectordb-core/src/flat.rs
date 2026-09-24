//! Exhaustive brute-force index: the exact ground truth every approximate
//! index (IVF, HNSW) is measured against. `recall@k = 1.0` by construction.

use std::collections::BTreeMap;

use crate::distance::{Distance, Metric};
use crate::index::IndexStrategy;
use crate::record::{Hit, Record};

pub struct FlatIndex {
    dimension: usize,
    metric: Metric,
    // BTreeMap (rather than HashMap) so that iteration order is
    // deterministic by id, which in turn makes `search` results
    // deterministic when several candidates tie on score.
    records: BTreeMap<u64, Vec<f32>>,
}

impl FlatIndex {
    pub fn new(dimension: usize, metric: Metric) -> Self {
        Self {
            dimension,
            metric,
            records: BTreeMap::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl IndexStrategy for FlatIndex {
    fn upsert(&mut self, record: Record) {
        assert_eq!(
            record.vector.len(),
            self.dimension,
            "record dimension does not match collection dimension"
        );
        self.records.insert(record.id, record.vector);
    }

    fn remove(&mut self, id: u64) -> bool {
        self.records.remove(&id).is_some()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(
            query.len(),
            self.dimension,
            "query dimension does not match collection dimension"
        );
        let mut hits: Vec<Hit> = self
            .records
            .iter()
            .map(|(&id, vector)| Hit {
                id,
                score: self.metric.distance(query, vector),
            })
            .collect();
        hits.sort_by(|a, b| a.score.total_cmp(&b.score));
        hits.truncate(k);
        hits
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index_with(pairs: &[(u64, [f32; 2])]) -> FlatIndex {
        let mut index = FlatIndex::new(2, Metric::L2);
        for &(id, vector) in pairs {
            index.upsert(Record::new(id, vector.to_vec()));
        }
        index
    }

    #[test]
    fn search_returns_closest_first() {
        let index = index_with(&[(1, [0.0, 0.0]), (2, [10.0, 10.0]), (3, [1.0, 1.0])]);
        let hits = index.search(&[0.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 3);
    }

    #[test]
    fn search_caps_at_available_records() {
        let index = index_with(&[(1, [0.0, 0.0])]);
        let hits = index.search(&[0.0, 0.0], 5);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn upsert_overwrites_existing_id() {
        let mut index = index_with(&[(1, [0.0, 0.0])]);
        index.upsert(Record::new(1, vec![10.0, 10.0]));
        assert_eq!(index.len(), 1);
        let hits = index.search(&[10.0, 10.0], 1);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn remove_deletes_record() {
        let mut index = index_with(&[(1, [0.0, 0.0]), (2, [1.0, 1.0])]);
        assert!(index.remove(1));
        assert!(!index.remove(1));
        assert_eq!(index.len(), 1);
        let hits = index.search(&[0.0, 0.0], 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 2);
    }

    #[test]
    #[should_panic]
    fn upsert_rejects_wrong_dimension() {
        let mut index = FlatIndex::new(3, Metric::L2);
        index.upsert(Record::new(1, vec![1.0, 2.0]));
    }
}
