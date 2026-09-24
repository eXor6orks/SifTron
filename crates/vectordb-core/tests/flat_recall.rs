//! `FlatIndex` is the ground truth against which every approximate index
//! (IVF, HNSW) will measure recall@k in later versions. These tests pin
//! down that its own results are always correct by definition: sorted,
//! duplicate-free, and identical to a naive re-implementation of k-NN.

use proptest::prelude::*;
use vectordb_core::{FlatIndex, IndexStrategy, Metric, Record};

fn naive_knn(records: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = records
        .iter()
        .map(|(id, v)| {
            let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (*id, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

proptest! {
    #[test]
    fn matches_naive_knn(
        records in prop::collection::vec((0u64..1000, prop::collection::vec(-10.0f32..10.0, 4)), 1..50)
            .prop_map(|mut v| { v.sort_by_key(|(id, _)| *id); v.dedup_by_key(|(id, _)| *id); v }),
        query in prop::collection::vec(-10.0f32..10.0, 4),
        k in 1usize..10,
    ) {
        let mut index = FlatIndex::new(4, Metric::L2);
        for (id, vector) in &records {
            index.upsert(Record::new(*id, vector.clone()));
        }

        let got: Vec<u64> = index.search(&query, k).into_iter().map(|h| h.id).collect();
        let expected = naive_knn(&records, &query, k);

        prop_assert_eq!(got, expected);
    }
}

#[test]
fn results_are_sorted_ascending() {
    let mut index = FlatIndex::new(2, Metric::L2);
    for i in 0..20u64 {
        index.upsert(Record::new(i, vec![i as f32, 0.0]));
    }
    let hits = index.search(&[0.0, 0.0], 10);
    for pair in hits.windows(2) {
        assert!(pair[0].score <= pair[1].score);
    }
}
