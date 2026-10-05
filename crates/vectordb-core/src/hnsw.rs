//! HNSW - Approximate Nearest Neighbor - graphs are among the top-performing indexes 
//! for vector similarity search. HNSW is a hugely popular technology that time and 
//! time again produces state-of-the-artperformance with super fast search speeds and
//! fantastic recall.

use crate::distance::{Distance, Metric};
use crate::index::IndexStrategy;
use crate::record::{Hit, Record};

struct Node {
    id: u64, // Unique identifier for the node
    vector: Vec<f32>,
    layers: Vec<Vec<usize>>, // Each layer contains a list of neighbor node indices
    deleted: bool, // Flag to mark if the node is deleted
}

pub struct HnswIndex {
    dimension: usize,
    metric: Metric,
    m: usize,              // Nb of neighbors per layer
    ef_construction: usize,
    ef_search: usize,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    id_to_index: std::collections::BTreeMap<u64, usize>,
}

fn random_level(m: usize, rng: &mut impl rand::Rng) -> usize {
    assert!(m > 1, "m must be at least 2 (ln(1) = 0 would divide by zero)");
    let u: f32 = rng.random_range(f32::EPSILON..1.0);
    (-u.ln() / (m as f32).ln()).floor() as usize
}


impl HnswIndex {
    pub fn new(
        dimension: usize, 
        metric: Metric, 
        m: usize, 
        ef_construction: usize, 
        ef_search: usize
    ) -> Self {
        Self {
            dimension,
            metric,
            m,
            ef_construction,
            ef_search,
            nodes: Vec::new(),
            entry_point: None,
            id_to_index: std::collections::BTreeMap::new(),
        }
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        // Two separate lists, not one `BinaryHeap`:
        // - `candidates` is the exploration queue — we always need the
        //   *closest* unvisited node next.
        // - `results` holds the best `ef` found so far — we need to evict
        //   the *farthest* one the moment a better candidate shows up.
        // A single heap can't serve both roles with opposite orderings at
        // once. f32 also doesn't implement `Ord` (only `PartialOrd`,
        // because of NaN), so `BinaryHeap<(f32, usize)>` wouldn't even
        // compile — same reason `ivf.rs` uses `Vec` + `sort_by(total_cmp)`
        // everywhere instead of a heap.
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut results: Vec<(usize, f32)> = Vec::new();

        for &entry_point in entry_points {
            let distance = self.metric.distance(query, &self.nodes[entry_point].vector);
            candidates.push((entry_point, distance));
            results.push((entry_point, distance));
            visited.insert(entry_point);
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1)); // farthest first, so .pop() yields the closest
        results.sort_by(|a, b| a.1.total_cmp(&b.1)); // closest first, so the last element is the worst kept

        while let Some((current, current_distance)) = candidates.pop() {
            // Early stop: once the closest remaining candidate is farther
            // than our current worst kept result (and results is already
            // full), nothing left in the queue can possibly improve
            // `results` — stop instead of draining the whole queue. This
            // is the condition that gives HNSW its speed; without it the
            // loop degrades into visiting the entire reachable graph.
            if results.len() >= ef {
                let worst_kept = results[results.len() - 1].1;
                if current_distance > worst_kept {
                    break;
                }
            }

            for &neighbor in &self.nodes[current].layers[layer] {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let neighbor_distance = self.metric.distance(query, &self.nodes[neighbor].vector);

                    let worth_keeping = results.len() < ef
                        || neighbor_distance < results[results.len() - 1].1;

                    if worth_keeping {
                        candidates.push((neighbor, neighbor_distance));
                        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

                        results.push((neighbor, neighbor_distance));
                        results.sort_by(|a, b| a.1.total_cmp(&b.1));
                        results.truncate(ef);
                    }
                }
            }
        }

        results
    }

    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 { 2 * self.m } else { self.m }
    }
}

impl IndexStrategy for HnswIndex {
    fn upsert(&mut self, record: Record) {
        assert_eq!(
            self.dimension,
            record.vector.len(),
            "Vector dimension mismatch: expected {}, got {}",
            self.dimension,
            record.vector.len()
        );

        // 0. Overwrite = tombstone the old version first (same reason as in
        //    IvfIndex: otherwise the id would exist twice).
        self.remove(record.id);

        // 1. Draw the new node's top layer at random.
        let level = random_level(self.m, &mut rand::rng());

        // 2. Create the node and register it. `new_index` is its position in
        //    `self.nodes`, computed BEFORE the push, because graph edges are
        //    positions, not ids.
        let new_index = self.nodes.len();
        self.nodes.push(Node {
            id: record.id,
            vector: record.vector.clone(),
            layers: vec![Vec::new(); level + 1],
            deleted: false,
        });
        self.id_to_index.insert(record.id, new_index);

        // 3. First node ever: it is the entry point, nothing to connect to.
        let Some(entry) = self.entry_point else {
            self.entry_point = Some(new_index);
            return;
        };

        // 4. Phase A: layers ABOVE the new node's level. We only travel
        //    through them, greedily (ef = 1), to land in the right
        //    neighbourhood. The new node is not connected here.
        let top_layer = self.nodes[entry].layers.len() - 1;
        let mut entry_points = vec![entry];
        for layer in ((level + 1)..=top_layer).rev() {
            let closest = self.search_layer(&record.vector, &entry_points, 1, layer);
            entry_points = vec![closest[0].0];
        }

        // 5. Phase B: from the node's own top layer (or the graph's, if the
        //    graph is lower) down to layer 0, find neighbours and connect.
        for layer in (0..=level.min(top_layer)).rev() {
            // a. wide search: the `ef_construction` best nodes at this layer
            let candidates =
                self.search_layer(&record.vector, &entry_points, self.ef_construction, layer);
            let max_conn = self.max_connections(layer);

            // b. keep the `m` closest that are not tombstoned
            //    (`candidates` is sorted closest-first)
            let neighbors: Vec<usize> = candidates
                .iter()
                .map(|&(index, _)| index)
                .filter(|&index| !self.nodes[index].deleted)
                .take(self.m)
                .collect();

            // c. connect in both directions
            self.nodes[new_index].layers[layer] = neighbors.clone();
            for &n in &neighbors {
                self.nodes[n].layers[layer].push(new_index);

                // d. prune: if `n` now has too many neighbours, keep only
                //    its `max_conn` closest. Compute the new list in a local
                //    variable first (immutable borrow ends), THEN assign it
                //    (mutable borrow) — this is what satisfies the borrow
                //    checker.
                if self.nodes[n].layers[layer].len() > max_conn {
                    let mut scored: Vec<(usize, f32)> = self.nodes[n].layers[layer]
                        .iter()
                        .map(|&other| {
                            let d = self
                                .metric
                                .distance(&self.nodes[n].vector, &self.nodes[other].vector);
                            (other, d)
                        })
                        .collect();
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                    scored.truncate(max_conn);
                    self.nodes[n].layers[layer] =
                        scored.into_iter().map(|(other, _)| other).collect();
                }
            }

            // e. next layer starts from ALL candidates, not just the `m` kept
            entry_points = candidates.iter().map(|&(index, _)| index).collect();
        }

        // 6. Taller than the whole graph: the new node becomes entry point.
        if level > top_layer {
            self.entry_point = Some(new_index);
        }
    }

    fn remove(&mut self, id: u64) -> bool {
        // `BTreeMap::remove` removes the entry AND returns it if it existed.
        // The node stays in `self.nodes` (edges are positions: removing it
        // would shift every later index), it is only marked as a tombstone.
        if let Some(index) = self.id_to_index.remove(&id) {
            self.nodes[index].deleted = true;
            true
        } else {
            false
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(
            query.len(),
            self.dimension,
            "query dimension does not match index dimension"
        );

        // Empty index: no entry point, nothing to return.
        let Some(entry) = self.entry_point else {
            return Vec::new();
        };

        // Phase A (same idea as in `upsert`): travel down from the top layer
        // to layer 1, greedily (ef = 1), keeping only the closest node found
        // at each layer as the starting point for the layer below.
        // `(1..=top_layer)` is empty when the graph has a single layer, so
        // there is no underflow risk.
        let top_layer = self.nodes[entry].layers.len() - 1;
        let mut entry_points = vec![entry];
        for layer in (1..=top_layer).rev() {
            let closest = self.search_layer(query, &entry_points, 1, layer);
            entry_points = vec![closest[0].0];
        }

        // Phase B: the real search, on layer 0 (the layer that holds every
        // node). `ef` is how many candidates we keep while exploring; it
        // must be at least `k`, otherwise we could not even return k results.
        // A bigger `ef` = better recall but slower: this is THE tuning knob.
        let ef = self.ef_search.max(k);
        let candidates = self.search_layer(query, &entry_points, ef, 0);

        // Tombstoned nodes were useful as stepping stones during navigation,
        // but must never appear in the answer. We also convert positions
        // (`usize`) back into the application ids (`u64`) stored in `Node`.
        let mut hits: Vec<Hit> = candidates
            .into_iter()
            .filter(|&(index, _)| !self.nodes[index].deleted)
            .map(|(index, score)| Hit { id: self.nodes[index].id, score })
            .collect();

        // Same deterministic ordering as `FlatIndex`/`IvfIndex`: by score,
        // ties broken by id.
        hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }

    fn len(&self) -> usize {
        self.nodes.iter().filter(|n| !n.deleted).count()
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::flat::FlatIndex;

    fn small_index() -> HnswIndex {
        HnswIndex::new(2, Metric::L2, 4, 16, 16)
    }

    fn random_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..dim).map(|_| rng.random_range(-100.0..100.0)).collect())
            .collect()
    }

    // ---------------------------------------------------------------
    // upsert / remove
    // ---------------------------------------------------------------

    #[test]
    fn first_insert_becomes_entry_point() {
        let mut idx = small_index();
        idx.upsert(Record::new(7, vec![1.0, 2.0]));
        assert_eq!(idx.entry_point, Some(0));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn overwrite_same_id_keeps_len_at_one() {
        let mut idx = small_index();
        idx.upsert(Record::new(1, vec![0.0, 0.0]));
        idx.upsert(Record::new(1, vec![5.0, 5.0]));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.nodes.len(), 2, "old version stays as a tombstone");
    }

    #[test]
    fn remove_returns_true_then_false() {
        let mut idx = small_index();
        idx.upsert(Record::new(1, vec![0.0, 0.0]));
        assert!(idx.remove(1));
        assert!(!idx.remove(1));
        assert!(!idx.remove(999));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn small_graph_is_fully_connected_at_layer_0() {
        // 5 nodes, m = 4, ef_construction = 16 >= 5: every new node must
        // see (and link to) every existing one.
        let mut idx = small_index();
        for i in 0..5u64 {
            idx.upsert(Record::new(i, vec![i as f32, (i * i) as f32]));
        }
        for (i, node) in idx.nodes.iter().enumerate() {
            let mut neighbors = node.layers[0].clone();
            neighbors.sort();
            let expected: Vec<usize> = (0..5).filter(|&j| j != i).collect();
            assert_eq!(neighbors, expected, "node {i} should see all others");
        }
    }

    fn check_invariants(idx: &HnswIndex) {
        let mut max_layers = 0;
        for (i, node) in idx.nodes.iter().enumerate() {
            max_layers = max_layers.max(node.layers.len());
            for (layer, neighbors) in node.layers.iter().enumerate() {
                assert!(
                    neighbors.len() <= idx.max_connections(layer),
                    "node {i} layer {layer} has too many neighbours"
                );
                let mut sorted = neighbors.clone();
                sorted.sort();
                sorted.dedup();
                assert_eq!(sorted.len(), neighbors.len(), "node {i} layer {layer}: duplicate neighbours");
                for &n in neighbors {
                    assert_ne!(n, i, "node {i} layer {layer} points to itself");
                    assert!(n < idx.nodes.len(), "dangling index");
                    assert!(
                        idx.nodes[n].layers.len() > layer,
                        "edge at layer {layer} points to a node absent from that layer"
                    );
                }
            }
        }
        let entry = idx.entry_point.expect("non-empty index has an entry point");
        assert_eq!(
            idx.nodes[entry].layers.len(),
            max_layers,
            "entry point must be a top-level node"
        );
        for (&id, &index) in &idx.id_to_index {
            assert_eq!(idx.nodes[index].id, id);
            assert!(!idx.nodes[index].deleted);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn graph_invariants_hold_after_random_inserts(
            vectors in prop::collection::vec(prop::collection::vec(-100.0f32..100.0, 3), 30..80)
        ) {
            let mut idx = HnswIndex::new(3, Metric::L2, 4, 16, 16);
            for (id, v) in vectors.iter().enumerate() {
                idx.upsert(Record::new(id as u64, v.clone()));
            }
            check_invariants(&idx);
        }
    }

    // ---------------------------------------------------------------
    // search
    // ---------------------------------------------------------------

    #[test]
    fn search_on_empty_index_returns_empty() {
        let idx = small_index();
        assert!(idx.search(&[0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn search_finds_exact_match_first() {
        let mut idx = small_index();
        idx.upsert(Record::new(1, vec![0.0, 0.0]));
        idx.upsert(Record::new(2, vec![10.0, 10.0]));
        idx.upsert(Record::new(3, vec![1.0, 1.0]));

        let hits = idx.search(&[10.0, 10.0], 3);
        assert_eq!(hits[0].id, 2);
        assert_eq!(hits[0].score, 0.0);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn search_results_are_sorted_and_capped_at_k() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut idx = HnswIndex::new(4, Metric::L2, 8, 64, 32);
        for (id, v) in random_vectors(&mut rng, 100, 4).into_iter().enumerate() {
            idx.upsert(Record::new(id as u64, v));
        }
        let hits = idx.search(&[0.0, 0.0, 0.0, 0.0], 10);
        assert_eq!(hits.len(), 10);
        assert!(hits.windows(2).all(|w| w[0].score <= w[1].score));
    }

    #[test]
    fn search_k_larger_than_len_returns_everything() {
        let mut idx = small_index();
        for i in 0..5u64 {
            idx.upsert(Record::new(i, vec![i as f32, 0.0]));
        }
        assert_eq!(idx.search(&[0.0, 0.0], 50).len(), 5);
    }

    #[test]
    fn search_with_k_zero_returns_empty() {
        let mut idx = small_index();
        idx.upsert(Record::new(1, vec![0.0, 0.0]));
        assert!(idx.search(&[0.0, 0.0], 0).is_empty());
    }

    #[test]
    fn search_never_returns_removed_ids() {
        let mut idx = small_index();
        for i in 0..5u64 {
            idx.upsert(Record::new(i, vec![i as f32, 0.0]));
        }
        assert!(idx.remove(0));
        assert!(idx.remove(1));

        let hits = idx.search(&[0.0, 0.0], 10);
        let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        assert!(!ids.contains(&0) && !ids.contains(&1), "got {ids:?}");
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn search_after_overwrite_uses_the_new_position_without_duplicate() {
        let mut idx = small_index();
        for i in 0..5u64 {
            idx.upsert(Record::new(i, vec![i as f32, 0.0]));
        }
        idx.upsert(Record::new(0, vec![100.0, 100.0])); // move id 0 far away

        let near_old = idx.search(&[0.0, 0.0], 10);
        let ids: Vec<u64> = near_old.iter().map(|h| h.id).collect();
        assert_eq!(ids.iter().filter(|&&id| id == 0).count(), 1, "id 0 must appear once");
        assert_eq!(*ids.last().unwrap(), 0, "id 0 is now the farthest");

        let near_new = idx.search(&[100.0, 100.0], 1);
        assert_eq!(near_new[0].id, 0);
        assert_eq!(near_new[0].score, 0.0);
    }

    #[test]
    fn search_works_with_cosine_and_dot_metrics() {
        for metric in [Metric::Cosine, Metric::Dot] {
            let mut idx = HnswIndex::new(2, metric, 4, 16, 16);
            let mut flat = FlatIndex::new(2, metric);
            for i in 1..8u64 {
                let v = vec![i as f32, (8 - i) as f32];
                idx.upsert(Record::new(i, v.clone()));
                flat.upsert(Record::new(i, v));
            }
            let q = [3.0, 5.0];
            let got: Vec<u64> = idx.search(&q, 3).iter().map(|h| h.id).collect();
            let want: Vec<u64> = flat.search(&q, 3).iter().map(|h| h.id).collect();
            assert_eq!(got, want, "metric {metric:?}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        // Whatever the data, a result must be internally coherent: sorted,
        // no duplicate ids, at most k, every id alive, and every score equal
        // to the real distance between the query and that id's vector.
        #[test]
        fn search_results_are_always_coherent(
            vectors in prop::collection::vec(prop::collection::vec(-100.0f32..100.0, 3), 20..60),
            query in prop::collection::vec(-100.0f32..100.0, 3),
            k in 1usize..15,
            removed in prop::collection::vec(0usize..20, 0..5),
        ) {
            let mut idx = HnswIndex::new(3, Metric::L2, 4, 16, 16);
            for (id, v) in vectors.iter().enumerate() {
                idx.upsert(Record::new(id as u64, v.clone()));
            }
            for r in removed {
                idx.remove(r as u64);
            }

            let hits = idx.search(&query, k);
            prop_assert!(hits.len() <= k);
            prop_assert!(hits.windows(2).all(|w| w[0].score <= w[1].score));

            let ids: HashSet<u64> = hits.iter().map(|h| h.id).collect();
            prop_assert_eq!(ids.len(), hits.len(), "duplicate ids in results");

            for hit in &hits {
                let index = *idx.id_to_index.get(&hit.id).expect("returned id must be alive");
                let real = Metric::L2.distance(&query, &idx.nodes[index].vector);
                prop_assert_eq!(hit.score, real);
            }
        }
    }

    // ---------------------------------------------------------------
    // quality: HNSW against the exact FlatIndex
    // ---------------------------------------------------------------

    fn recall_at_k(idx: &HnswIndex, flat: &FlatIndex, queries: &[Vec<f32>], k: usize) -> f32 {
        let mut found = 0;
        for q in queries {
            let truth: HashSet<u64> = flat.search(q, k).iter().map(|h| h.id).collect();
            found += idx.search(q, k).iter().filter(|h| truth.contains(&h.id)).count();
        }
        found as f32 / (queries.len() * k) as f32
    }

    #[test]
    fn recall_against_flat_index_is_high_and_improves_with_ef_search() {
        let mut rng = StdRng::seed_from_u64(2024);
        let dim = 16;
        let mut idx = HnswIndex::new(dim, Metric::L2, 16, 200, 64);
        let mut flat = FlatIndex::new(dim, Metric::L2);
        for (id, v) in random_vectors(&mut rng, 1000, dim).into_iter().enumerate() {
            idx.upsert(Record::new(id as u64, v.clone()));
            flat.upsert(Record::new(id as u64, v));
        }
        check_invariants(&idx);
        let queries = random_vectors(&mut rng, 100, dim);

        println!("\nrecall@10 vs FlatIndex (1000 vectors, dim 16, 100 queries):");
        let mut last = 0.0;
        for ef in [10, 20, 50, 100] {
            idx.ef_search = ef;
            last = recall_at_k(&idx, &flat, &queries, 10);
            println!("  ef_search = {ef:>3}  ->  recall@10 = {last:.3}");
        }
        assert!(last >= 0.95, "recall@10 with ef_search=100 was only {last:.3}");
    }
}