//! Compares FlatIndex, IvfIndex and HnswIndex on the same dataset:
//! quality (recall@k) against speed (queries per second).
//!
//! Run with:   cargo bench --bench compare

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use vectordb_core::{FlatIndex, HnswIndex, IndexStrategy, IvfIndex, Metric, Record};

const N: usize = 10_000; // vectors stored in each index
const DIM: usize = 64;
const N_CLUSTERS: usize = 50; // how many "topics" the fake data has
const N_QUERIES: usize = 200;
const K: usize = 10;

/// Clustered data, a bit like real embeddings: points gather around a few
/// random centres (pure uniform noise would have no structure for IVF to use).
/// Returns `total` points; the last ones are held out as queries.
fn make_dataset(rng: &mut StdRng, total: usize) -> Vec<Vec<f32>> {
    let centres: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.random_range(-100.0..100.0)).collect())
        .collect();
    (0..total)
        .map(|_| {
            let centre = &centres[rng.random_range(0..N_CLUSTERS)];
            centre.iter().map(|c| c + rng.random_range(-8.0..8.0)).collect()
        })
        .collect()
}

/// Returns (recall@K, queries per second) for one index in its current setting.
fn measure(index: &impl IndexStrategy, queries: &[Vec<f32>], truth: &[HashSet<u64>]) -> (f32, f64) {
    // Quality: how many of the true K nearest did we find?
    let mut found = 0;
    for (query, true_ids) in queries.iter().zip(truth) {
        found += index.search(query, K).iter().filter(|h| true_ids.contains(&h.id)).count();
    }
    let recall = found as f32 / (queries.len() * K) as f32;

    // Speed: best of 3 passes over all queries (less noisy than a single pass).
    let mut best = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        for query in queries {
            // black_box: stop the compiler from optimising the work away
            black_box(index.search(black_box(query), K));
        }
        best = best.min(start.elapsed().as_secs_f64());
    }
    (recall, queries.len() as f64 / best)
}

fn row(name: &str, setting: &str, recall: f32, qps: f64, flat_qps: f64) {
    println!("{name:<6} {setting:<14} {recall:>9.3} {qps:>12.0} {:>9.1}x", qps / flat_qps);
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let data = make_dataset(&mut rng, N + N_QUERIES);
    let (stored, queries) = data.split_at(N);
    let queries = queries.to_vec();

    println!("{N} vectors, dim {DIM}, {N_QUERIES} queries, k = {K}, metric = L2\n");

    // ---- Flat: exact, gives the ground truth ----
    let t = Instant::now();
    let mut flat = FlatIndex::new(DIM, Metric::L2);
    for (id, v) in stored.iter().enumerate() {
        flat.upsert(Record::new(id as u64, v.clone()));
    }
    println!("build  Flat: {:>7.2}s", t.elapsed().as_secs_f64());
    let truth: Vec<HashSet<u64>> = queries
        .iter()
        .map(|q| flat.search(q, K).iter().map(|h| h.id).collect())
        .collect();

    // ---- IVF: nlist ~ sqrt(N); trained once, on all the data ----
    let t = Instant::now();
    let nlist = 100;
    let mut ivf = IvfIndex::new(DIM, Metric::L2, nlist, N, 1);
    for (id, v) in stored.iter().enumerate() {
        ivf.upsert(Record::new(id as u64, v.clone()));
    }
    println!("build  IVF : {:>7.2}s (nlist = {nlist})", t.elapsed().as_secs_f64());

    // ---- HNSW ----
    let t = Instant::now();
    let mut hnsw = HnswIndex::new(DIM, Metric::L2, 16, 200, 10);
    for (id, v) in stored.iter().enumerate() {
        hnsw.upsert(Record::new(id as u64, v.clone()));
    }
    println!("build  HNSW: {:>7.2}s (m = 16, ef_construction = 200)\n", t.elapsed().as_secs_f64());

    println!("{:<6} {:<14} {:>9} {:>12} {:>10}", "index", "setting", "recall@10", "queries/s", "vs Flat");
    println!("{}", "-".repeat(55));

    let (r, flat_qps) = measure(&flat, &queries, &truth);
    row("Flat", "-", r, flat_qps, flat_qps);

    for nprobe in [1, 2, 4, 8, 16, 32] {
        ivf.set_nprobe(nprobe);
        let (r, qps) = measure(&ivf, &queries, &truth);
        row("IVF", &format!("nprobe={nprobe}"), r, qps, flat_qps);
    }

    for ef in [10, 20, 50, 100, 200] {
        hnsw.set_ef_search(ef);
        let (r, qps) = measure(&hnsw, &queries, &truth);
        row("HNSW", &format!("ef_search={ef}"), r, qps, flat_qps);
    }
}