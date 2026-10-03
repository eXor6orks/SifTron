//! HNSW - Approximate Nearest Neighbor - graphs are among the top-performing indexes 
//! for vector similarity search. HNSW is a hugely popular technology that time and 
//! time again produces state-of-the-artperformance with super fast search speeds and
//! fantastic recall.

use crate::distance::{Distance, Metric};
use crate::index::IndexStrategy;
use crate::record::{Hit, Record};

struct Node {
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
}

impl IndexStrategy for HnswIndex {
    fn upsert(&mut self, _record: Record) {
        // Implementation for inserting a record into the HNSW index
        // This would involve creating a new node, determining its level,
        // and connecting it to existing nodes in the graph.
        todo!()
    }

    fn remove(&mut self, _id: u64) -> bool {
        todo!()
    }

    fn search(&self, _query: &[f32], _k: usize) -> Vec<Hit> {
        todo!()
    }

    fn len(&self) -> usize {
        self.nodes.iter().filter(|n| !n.deleted).count()
    }
}