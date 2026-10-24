//! Flat index with scalar (SQ8) quantization: each component is stored on one
//! byte (4x less memory than `FlatIndex`) at the cost of a small precision loss.
//!
//! The quantizer needs data to learn per-dimension ranges, so the index works
//! in two phases: until `train_threshold` records have been inserted it keeps
//! the raw vectors in `pending` (exact search); once the threshold is reached
//! the quantizer is trained on them, they are encoded, and every later insert
//! is encoded on the fly.

use std::collections::BTreeMap;

use crate::distance::{Distance, Metric};
use crate::index::IndexStrategy;
use crate::quantize::ScalarQuantizer;
use crate::record::{Hit, Record};

pub struct FlatSq8Index {
    dimension: usize,
    metric: Metric,
    train_threshold: usize,
    quantizer: Option<ScalarQuantizer>,  // None tant que non entraîné
    codes: BTreeMap<u64, Vec<u8>>,       // utilisé après entraînement
    pending: BTreeMap<u64, Vec<f32>>,    // utilisé avant entraînement
}

impl FlatSq8Index {
    pub fn new(
        dimension: usize,
        metric: Metric,
        train_threshold: usize
    ) -> Self {
        assert!(train_threshold > 0, "train_threshold must be at least 1");
        FlatSq8Index {
            dimension,
            metric,
            train_threshold,
            quantizer: None,
            codes: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Whether the quantizer has been trained (vectors are stored as codes).
    pub fn is_trained(&self) -> bool {
        self.quantizer.is_some()
    }

    /// Trains the quantizer on the pending vectors and encodes them.
    fn train(&mut self) {
        let training: Vec<Vec<f32>> = self.pending.values().cloned().collect();
        let quantizer = ScalarQuantizer::train(&training);
        for (id, vector) in std::mem::take(&mut self.pending) {
            self.codes.insert(id, quantizer.encode(&vector));
        }
        self.quantizer = Some(quantizer);
    }
}

impl IndexStrategy for FlatSq8Index {
    fn upsert(&mut self, record: Record) {
        assert_eq!(
            record.vector.len(),
            self.dimension,
            "record dimension does not match collection dimension"
        );
        match &self.quantizer {
            Some(quantizer) => {
                self.codes.insert(record.id, quantizer.encode(&record.vector));
            }
            None => {
                self.pending.insert(record.id, record.vector);
                if self.pending.len() >= self.train_threshold {
                    self.train();
                }
            }
        }
    }

    fn remove(&mut self, id: u64) -> bool {
        self.codes.remove(&id).is_some() || self.pending.remove(&id).is_some()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(
            query.len(),
            self.dimension,
            "query dimension does not match collection dimension"
        );
        let mut hits: Vec<Hit> = match &self.quantizer {
            Some(quantizer) => self
                .codes
                .iter()
                .map(|(&id, code)| Hit {
                    id,
                    score: quantizer.distance(&self.metric, query, code),
                })
                .collect(),
            None => self
                .pending
                .iter()
                .map(|(&id, vector)| Hit {
                    id,
                    score: self.metric.distance(query, vector),
                })
                .collect(),
        };
        hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }

    fn len(&self) -> usize {
        self.codes.len() + self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;

    fn vec_for(i: u64) -> Vec<f32> {
        vec![i as f32, (i * 7 % 13) as f32, (i % 5) as f32]
    }

    #[test]
    fn stays_untrained_below_threshold() {
        let mut index = FlatSq8Index::new(3, Metric::L2, 5);
        for i in 0..4 {
            index.upsert(Record::new(i, vec_for(i)));
        }
        assert!(!index.is_trained());
        assert_eq!(index.len(), 4);
        // exact search while pending
        let hits = index.search(&vec_for(2), 1);
        assert_eq!(hits[0].id, 2);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn trains_when_threshold_is_reached() {
        let mut index = FlatSq8Index::new(3, Metric::L2, 5);
        for i in 0..5 {
            index.upsert(Record::new(i, vec_for(i)));
        }
        assert!(index.is_trained());
        assert!(index.pending.is_empty());
        assert_eq!(index.len(), 5);
        index.upsert(Record::new(99, vec_for(3)));
        assert_eq!(index.len(), 6);
    }

    #[test]
    fn upsert_overwrites_in_both_phases() {
        let mut index = FlatSq8Index::new(3, Metric::L2, 3);
        index.upsert(Record::new(1, vec_for(1)));
        index.upsert(Record::new(1, vec_for(2)));
        assert_eq!(index.len(), 1);
        for i in 2..5 {
            index.upsert(Record::new(i, vec_for(i)));
        }
        assert!(index.is_trained());
        let before = index.len();
        index.upsert(Record::new(2, vec_for(4)));
        assert_eq!(index.len(), before);
    }

    #[test]
    fn remove_works_before_and_after_training() {
        let mut index = FlatSq8Index::new(3, Metric::L2, 3);
        index.upsert(Record::new(1, vec_for(1)));
        assert!(index.remove(1));
        assert!(!index.remove(1));
        for i in 0..3 {
            index.upsert(Record::new(i, vec_for(i)));
        }
        assert!(index.is_trained());
        assert!(index.remove(0));
        assert!(!index.remove(0));
        assert_eq!(index.len(), 2);
        assert!(index.search(&vec_for(0), 10).iter().all(|h| h.id != 0));
    }

    #[test]
    fn search_is_sorted_and_capped() {
        let mut index = FlatSq8Index::new(3, Metric::L2, 10);
        for i in 0..20 {
            index.upsert(Record::new(i, vec_for(i)));
        }
        let hits = index.search(&vec_for(7), 5);
        assert_eq!(hits.len(), 5);
        assert!(hits.windows(2).all(|w| w[0].score <= w[1].score));
        assert_eq!(index.search(&vec_for(7), 100).len(), 20);
    }

    #[test]
    fn recall_against_flat_index_on_clustered_data() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::collections::HashSet;

        let mut rng = StdRng::seed_from_u64(1);
        let (dim, n, n_queries, n_clusters, k) = (64, 3000, 100, 30, 10);
        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random_range(-100.0..100.0)).collect())
            .collect();
        let points: Vec<Vec<f32>> = (0..n + n_queries)
            .map(|_| {
                let c = &centres[rng.random_range(0..n_clusters)];
                c.iter().map(|x| x + rng.random_range(-8.0..8.0)).collect()
            })
            .collect();
        let (stored, queries) = points.split_at(n);

        let mut sq8 = FlatSq8Index::new(dim, Metric::L2, 1000);
        let mut flat = FlatIndex::new(dim, Metric::L2);
        for (id, v) in stored.iter().enumerate() {
            sq8.upsert(Record::new(id as u64, v.clone()));
            flat.upsert(Record::new(id as u64, v.clone()));
        }

        let mut found = 0;
        for q in queries {
            let truth: HashSet<u64> = flat.search(q, k).iter().map(|h| h.id).collect();
            found += sq8.search(q, k).iter().filter(|h| truth.contains(&h.id)).count();
        }
        let recall = found as f32 / (n_queries * k) as f32;
        println!("FlatSq8 recall@{k} = {recall:.3}");
        assert!(recall >= 0.90, "recall@{k} was only {recall:.3}");
    }

    #[test]
    #[should_panic]
    fn upsert_rejects_wrong_dimension() {
        let mut index = FlatSq8Index::new(3, Metric::L2, 5);
        index.upsert(Record::new(1, vec![1.0, 2.0]));
    }
}
