//! IVF (Inverted File) index: routes each vector to its nearest of `nlist`
//! centroids (computed by [`crate::kmeans`]), so that a search only needs
//! to scan the `nprobe` most promising clusters instead of every vector.

use std::collections::BTreeMap;

use crate::distance::{Distance, Metric};
use crate::index::IndexStrategy;
use crate::kmeans::kmeans;
use crate::record::{Hit, Record};

pub struct IvfIndex {
    dimension: usize,
    metric: Metric,
    nlist: usize,
    train_threshold: usize,
    nprobe: usize,

    // These three fields together encode a small state machine:
    // - `centroids` empty  → untrained, everything lives in `pending`.
    // - `centroids` non-empty → trained, `pending` is always empty,
    //   everything lives in `clusters`.
    // We never mix the two states, which is what keeps `is_trained()`
    // a single, cheap, unambiguous check.
    /// Empty until `train()` has run.
    centroids: Vec<Vec<f32>>,
    /// `clusters[j]` holds the vectors routed to `centroids[j]`.
    clusters: Vec<BTreeMap<u64, Vec<f32>>>,
    /// Vectors inserted before training has happened yet.
    pending: BTreeMap<u64, Vec<f32>>,
}

impl IvfIndex {
    /// # Panics
    /// Panics if `nlist == 0`, `nprobe == 0`, or `train_threshold < nlist`
    /// (training needs at least one candidate vector per cluster) —
    /// checked eagerly here so a misconfiguration fails at construction
    /// time rather than deep inside `kmeans` once the buffer fills.
    pub fn new(
        dimension: usize,
        metric: Metric,
        nlist: usize,
        train_threshold: usize,
        nprobe: usize,
    ) -> Self {
        // We validate the *relationship* between nlist and train_threshold
        // here, not just that each is individually non-zero. Without this,
        // a caller could set train_threshold=5 with nlist=10, and the
        // crash would happen much later, inside kmeans(), the moment the
        // 5th vector triggers training — a far more confusing place to
        // discover the mistake than right here at construction.
        assert!(nlist > 0, "nlist must be at least 1");
        assert!(nprobe > 0, "nprobe must be at least 1");
        assert!(
            train_threshold >= nlist,
            "train_threshold ({train_threshold}) must be at least nlist ({nlist})"
        );
        Self {
            dimension,
            metric,
            nlist,
            train_threshold,
            nprobe,
            centroids: Vec::new(),
            clusters: Vec::new(),
            pending: BTreeMap::new(),
        }
    }

    pub fn is_trained(&self) -> bool {
        !self.centroids.is_empty()
    }

    /// Runs k-means once on everything accumulated in `pending`, then
    /// empties it into `clusters`. Called automatically by `upsert` the
    /// moment `pending` reaches `train_threshold` — see the state machine
    /// note on the struct fields above.
    fn train(&mut self) {
        // `kmeans()` wants two parallel slices: the vectors, and (after the
        // call) which cluster each one landed in, in the same order. We
        // build `ids` and `vectors` from a *single* pass over `pending`
        // with `.unzip()`, rather than calling `.keys()` and `.values()`
        // separately — that guarantees element i of each Vec really does
        // come from the same (id, vector) pair, with no risk of the two
        // iterators silently drifting out of sync.
        let (ids, vectors): (Vec<u64>, Vec<Vec<f32>>) =
            self.pending.iter().map(|(&id, v)| (id, v.clone())).unzip();

        // Defensive clamp: `new()` already guarantees train_threshold >=
        // nlist, so vectors.len() >= nlist should always hold here. This
        // line is a safety net, not something normal operation relies on —
        // a `kmeans` panic from a k > vectors.len() mismatch would be far
        // harder to diagnose from inside `train()` than a quiet clamp.
        let effective_nlist = self.nlist.min(vectors.len());

        // train() doesn't take a seeded RNG parameter — unlike kmeans()
        // itself, which does for testability (see kmeans.rs). That's a
        // deliberate difference: IvfIndex is exercised through the public
        // IndexStrategy trait, whose upsert/search signatures are fixed —
        // there's no seed to thread through. Tests instead check *recall
        // properties* (e.g. nprobe == nlist must match FlatIndex exactly)
        // rather than exact clustering equality, so non-determinism here
        // is fine.
        let mut rng = rand::rng();
        let result = kmeans(&vectors, effective_nlist, &self.metric, 25, &mut rng);

        // One empty BTreeMap per centroid, then dispatch every (id, vector)
        // into the cluster kmeans assigned it to.
        self.clusters = vec![BTreeMap::new(); result.centroids.len()];
        for ((id, vector), cluster) in ids.into_iter().zip(vectors).zip(result.assignments) {
            self.clusters[cluster].insert(id, vector);
        }
        self.centroids = result.centroids;
        self.pending.clear(); // now trained: pending must go back to empty
    }

    // Index of the single closest centroid — used to route one vector at
    /// insert time. Not used at search time; see `nearest_clusters` below.
    fn nearest_cluster(&self, vector: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j, self.metric.distance(vector, c)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(j, _)| j)
            .expect("centroids is non-empty (only called once trained)")
    }

    /// Indices of the `nprobe` centroids closest to `vector` (fewer if
    /// `nprobe` exceeds `nlist` — `.truncate` silently handles that rather
    /// than needing a separate branch, since truncating to a length larger
    /// than the Vec is a no-op in Rust).
    ///
    /// This is a *different* function from `nearest_cluster` above, not a
    /// generalisation reused with n=1, because insertion only ever needs
    /// the single best cluster (an O(nlist) scan is enough), while search
    /// needs the top-`nprobe`, which means sorting all of them. Keeping
    /// them separate keeps the hot insert path from paying for a sort it
    /// doesn't need.
    fn nearest_clusters(&self, vector: &[f32]) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j, self.metric.distance(vector, c)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(self.nprobe.min(self.centroids.len()));
        scored.into_iter().map(|(j, _)| j).collect()
    }

}

// These two free functions aren't methods on IvfIndex because they don't
// need `self` — they operate on whichever BTreeMap they're handed. That
// means the exact same code scores the `pending` buffer (untrained path)
// and each individual cluster (trained path) — one implementation, reused
// rather than duplicated, matching the "score a bag of (id, vector) pairs"
// operation FlatIndex also does internally.
fn score_all(records: &BTreeMap<u64, Vec<f32>>, query: &[f32], metric: &Metric) -> Vec<Hit> {
    records
        .iter()
        .map(|(&id, vector)| Hit { id, score: metric.distance(query, vector) })
        .collect()
}

fn top_k(mut hits: Vec<Hit>, k: usize) -> Vec<Hit> {
    // Sorting by score alone isn't quite enough: when several hits tie
    // exactly (a real case once you probe multiple clusters — a query
    // equidistant from two points in *different* clusters), relying only
    // on Vec::sort_by's stability breaks down, because the *order hits
    // were collected in* differs from FlatIndex's simple single-BTreeMap
    // scan. Breaking ties by id gives every backend the same, fully
    // deterministic ordering regardless of how it internally gathered
    // candidates.
    hits.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.id.cmp(&b.id)));
    hits.truncate(k);
    hits
}

impl IndexStrategy for IvfIndex {
    fn upsert(&mut self, record: Record) {
        assert_eq!(
            record.vector.len(),
            self.dimension,
            "record dimension does not match collection dimension"
        );

        // Idempotent overwrite: if this id already exists somewhere (in
        // `pending`, or already routed to some cluster), we must drop that
        // old copy *before* deciding where the new vector goes. Otherwise,
        // re-inserting under a (possibly different) nearest cluster would
        // leave a stale duplicate behind in whichever cluster it used to
        // belong to — the id would then show up twice in future searches.
        self.remove(record.id);

        if self.is_trained() {
            let cluster = self.nearest_cluster(&record.vector);
            self.clusters[cluster].insert(record.id, record.vector);
        } else {
            self.pending.insert(record.id, record.vector);
            // The auto-training trigger lives here, and only here: it's
            // the one place that ever transitions the state machine from
            // "untrained" to "trained".
            if self.pending.len() >= self.train_threshold {
                self.train();
            }
        }
    }

    fn remove(&mut self, id: u64) -> bool {
        // Check pending first — cheap, and covers the untrained case
        // entirely on its own (the loop below is simply empty then).
        if self.pending.remove(&id).is_some() {
            return true;
        }
        // Linear scan over clusters: we don't track which cluster an id
        // lives in separately, so removal costs O(nlist) map lookups.
        // Acceptable while nlist stays small (a few hundred); if this ever
        // becomes a hot path at larger scale, the fix would be a parallel
        // `id -> cluster index` map, not a rewrite of this logic.
        for cluster in &mut self.clusters {
            if cluster.remove(&id).is_some() {
                return true;
            }
        }
        false
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(
            query.len(),
            self.dimension,
            "query dimension does not match collection dimension"
        );

        // Untrained fallback: behave exactly like FlatIndex over whatever
        // is in `pending`. This is what makes the index usable from the
        // very first insert, before there's enough data to cluster at all.
        if !self.is_trained() {
            return top_k(score_all(&self.pending, query, &self.metric), k);
        }

        // Trained path: only look inside the nprobe closest clusters,
        // rather than every vector — this is the entire point of IVF.
        let mut hits = Vec::new();
        for j in self.nearest_clusters(query) {
            hits.extend(score_all(&self.clusters[j], query, &self.metric));
        }
        top_k(hits, k)
    }

    fn len(&self) -> usize {
        self.pending.len() + self.clusters.iter().map(BTreeMap::len).sum::<usize>()
    }
}


/// Test zone
/// 

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;

    use super::*;
    use crate::flat::FlatIndex;

    fn blob_dataset() -> Vec<(u64, Vec<f32>)> {
        let centers = [[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0]];
        let offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]];
        let mut id = 0u64;
        let mut points = Vec::new();
        for c in centers {
            for o in offsets {
                points.push((id, vec![c[0] + o[0], c[1] + o[1]]));
                id += 1;
            }
        }
        points
    }

    #[test]
    fn search_before_training_is_exact() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 100, 1);
        ivf.upsert(Record::new(1, vec![0.0, 0.0]));
        ivf.upsert(Record::new(2, vec![10.0, 10.0]));
        ivf.upsert(Record::new(3, vec![1.0, 1.0]));

        assert!(!ivf.is_trained());
        let hits = ivf.search(&[0.0, 0.0], 2);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 3);
    }

    #[test]
    fn training_triggers_automatically_at_threshold() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());
        assert_eq!(ivf.len(), 24);
    }

    #[test]
    fn nprobe_equal_to_nlist_matches_flat_exactly() {
        let dataset = blob_dataset();
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        let mut flat = FlatIndex::new(2, Metric::L2);
        for (id, vector) in &dataset {
            ivf.upsert(Record::new(*id, vector.clone()));
            flat.upsert(Record::new(*id, vector.clone()));
        }
        assert!(ivf.is_trained());

        for query in [[0.5, 0.5], [50.0, 50.0], [25.0, 25.0]] {
            let ivf_hits = ivf.search(&query, 5);
            let flat_hits = flat.search(&query, 5);
            let ivf_ids: Vec<u64> = ivf_hits.iter().map(|h| h.id).collect();
            let flat_ids: Vec<u64> = flat_hits.iter().map(|h| h.id).collect();
            assert_eq!(ivf_ids, flat_ids, "query {query:?}: IVF must match Flat when nprobe == nlist");
        }
    }

    #[test]
    fn upsert_after_training_is_findable() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());

        ivf.upsert(Record::new(999, vec![25.0, 25.0]));
        let hits = ivf.search(&[25.0, 25.0], 1);
        assert_eq!(hits[0].id, 999);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn remove_works_before_and_after_training() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());
        let total = ivf.len();

        assert!(ivf.remove(0));
        assert!(!ivf.remove(0));
        assert_eq!(ivf.len(), total - 1);
    }

    #[test]
    fn nprobe_larger_than_nlist_does_not_panic() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 999);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        let hits = ivf.search(&[0.0, 0.0], 3);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    #[should_panic]
    fn rejects_train_threshold_below_nlist() {
        IvfIndex::new(2, Metric::L2, 10, 5, 1);
    }

    #[test]
    #[should_panic]
    fn upsert_rejects_wrong_dimension() {
        let mut ivf = IvfIndex::new(3, Metric::L2, 2, 10, 1);
        ivf.upsert(Record::new(1, vec![1.0, 2.0]));
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let ivf = IvfIndex::new(2, Metric::L2, 4, 100, 1);
        assert!(!ivf.is_trained());
        assert_eq!(ivf.len(), 0);
        assert!(ivf.search(&[0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn upsert_overwrite_before_training_does_not_duplicate() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 100, 1);
        ivf.upsert(Record::new(1, vec![0.0, 0.0]));
        ivf.upsert(Record::new(1, vec![10.0, 10.0]));

        assert!(!ivf.is_trained());
        assert_eq!(ivf.len(), 1);
        let hits = ivf.search(&[10.0, 10.0], 1);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn upsert_overwrite_after_training_relocates_without_duplicate() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());
        let total = ivf.len();

        // id 0 originally lives in the (0,0) blob; move it into the (50,50) blob.
        ivf.upsert(Record::new(0, vec![50.0, 50.0]));
        assert_eq!(ivf.len(), total, "overwrite must not change the total count");

        let hits = ivf.search(&[50.0, 50.0], 1);
        assert_eq!(hits[0].id, 0);
        assert_eq!(hits[0].score, 0.0);

        // It must be gone from its old location too, not duplicated.
        let old_hits = ivf.search(&[0.0, 0.0], 10);
        assert!(!old_hits.iter().any(|h| h.id == 0));
    }

    #[test]
    fn remove_nonexistent_id_returns_false_before_and_after_training() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        assert!(!ivf.remove(42));

        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());
        assert!(!ivf.remove(9999));
    }

    #[test]
    fn remove_all_then_reinsert_after_training_still_searchable() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());

        for id in 0..24u64 {
            assert!(ivf.remove(id));
        }
        assert_eq!(ivf.len(), 0);
        assert!(
            ivf.is_trained(),
            "training state persists even once all data has been removed"
        );

        ivf.upsert(Record::new(100, vec![0.0, 0.0]));
        let hits = ivf.search(&[0.0, 0.0], 1);
        assert_eq!(hits[0].id, 100);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn search_k_larger_than_len_returns_all_before_training() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 100, 1);
        ivf.upsert(Record::new(1, vec![0.0, 0.0]));
        ivf.upsert(Record::new(2, vec![1.0, 1.0]));

        let hits = ivf.search(&[0.0, 0.0], 50);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn search_k_larger_than_len_returns_all_after_training() {
        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in blob_dataset() {
            ivf.upsert(Record::new(id, vector));
        }
        assert!(ivf.is_trained());

        let hits = ivf.search(&[0.0, 0.0], 1000);
        assert_eq!(hits.len(), ivf.len());
    }

    #[test]
    fn nlist_one_is_equivalent_to_a_full_flat_scan() {
        let dataset = blob_dataset();
        let mut ivf = IvfIndex::new(2, Metric::L2, 1, 24, 1);
        let mut flat = FlatIndex::new(2, Metric::L2);
        for (id, vector) in &dataset {
            ivf.upsert(Record::new(*id, vector.clone()));
            flat.upsert(Record::new(*id, vector.clone()));
        }
        assert!(ivf.is_trained());

        for query in [[0.5, 0.5], [50.0, 50.0], [25.0, 25.0]] {
            let ivf_ids: Vec<u64> = ivf.search(&query, 6).iter().map(|h| h.id).collect();
            let flat_ids: Vec<u64> = flat.search(&query, 6).iter().map(|h| h.id).collect();
            assert_eq!(ivf_ids, flat_ids, "query {query:?}");
        }
    }

    #[test]
    fn nprobe_equal_to_nlist_matches_flat_for_cosine_metric() {
        let dataset = blob_dataset();
        let mut ivf = IvfIndex::new(2, Metric::Cosine, 4, 24, 4);
        let mut flat = FlatIndex::new(2, Metric::Cosine);
        for (id, vector) in &dataset {
            ivf.upsert(Record::new(*id, vector.clone()));
            flat.upsert(Record::new(*id, vector.clone()));
        }
        assert!(ivf.is_trained());

        for query in [[0.5, 0.5], [50.0, 50.0], [25.0, 25.0]] {
            let ivf_ids: Vec<u64> = ivf.search(&query, 5).iter().map(|h| h.id).collect();
            let flat_ids: Vec<u64> = flat.search(&query, 5).iter().map(|h| h.id).collect();
            assert_eq!(ivf_ids, flat_ids, "query {query:?}: probing every cluster must equal a full scan");
        }
    }

    #[test]
    fn nprobe_equal_to_nlist_matches_flat_for_dot_metric() {
        let dataset = blob_dataset();
        let mut ivf = IvfIndex::new(2, Metric::Dot, 4, 24, 4);
        let mut flat = FlatIndex::new(2, Metric::Dot);
        for (id, vector) in &dataset {
            ivf.upsert(Record::new(*id, vector.clone()));
            flat.upsert(Record::new(*id, vector.clone()));
        }
        assert!(ivf.is_trained());

        for query in [[0.5, 0.5], [50.0, 50.0], [25.0, 25.0]] {
            let ivf_ids: Vec<u64> = ivf.search(&query, 5).iter().map(|h| h.id).collect();
            let flat_ids: Vec<u64> = flat.search(&query, 5).iter().map(|h| h.id).collect();
            assert_eq!(ivf_ids, flat_ids, "query {query:?}: probing every cluster must equal a full scan");
        }
    }

    #[test]
    fn increasing_nprobe_never_decreases_recall() {
        // `nearest_clusters` always probes the top-`nprobe` centroids by
        // distance, so the set of clusters scanned at nprobe=k is a subset
        // of the set scanned at nprobe=k+1. That makes the candidate pool
        // (and therefore recall against the true top-k) monotonically
        // non-decreasing in nprobe *by construction*, for any fixed
        // clustering. To test that directly, this trains a single index
        // once (training is unseeded, see `train()`) and reaches into its
        // private `centroids`/`clusters` to replay the search at several
        // nprobe values against that one fixed clustering, rather than
        // retraining (and so re-clustering, possibly differently) per
        // nprobe like calling the public `search` on separately
        // constructed indexes would.
        let dataset = blob_dataset();
        let mut flat = FlatIndex::new(2, Metric::L2);
        for (id, vector) in &dataset {
            flat.upsert(Record::new(*id, vector.clone()));
        }

        let mut ivf = IvfIndex::new(2, Metric::L2, 4, 24, 4);
        for (id, vector) in &dataset {
            ivf.upsert(Record::new(*id, vector.clone()));
        }
        assert!(ivf.is_trained());

        let query = [20.0, 0.0];
        let flat_top8: HashSet<u64> = flat.search(&query, 8).iter().map(|h| h.id).collect();

        let overlap = |nprobe: usize| -> usize {
            let mut scored: Vec<(usize, f32)> = ivf
                .centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, Metric::L2.distance(&query, c)))
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            scored.truncate(nprobe);

            let mut hits = Vec::new();
            for (j, _) in scored {
                hits.extend(score_all(&ivf.clusters[j], &query, &Metric::L2));
            }
            top_k(hits, 8)
                .iter()
                .filter(|h| flat_top8.contains(&h.id))
                .count()
        };

        let overlap_1 = overlap(1);
        let overlap_2 = overlap(2);
        let overlap_3 = overlap(3);
        let overlap_4 = overlap(4);

        assert!(overlap_1 <= overlap_2);
        assert!(overlap_2 <= overlap_3);
        assert!(overlap_3 <= overlap_4);
        assert_eq!(overlap_4, 8, "probing every cluster must recover the exact top-8");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        // With nprobe == nlist, IVF always scans every cluster, i.e. every
        // vector — so regardless of how kmeans happened to partition this
        // particular random dataset, results must be identical to a full
        // FlatIndex scan. This holds for any input, which is why it's
        // checked as a property rather than against a handful of fixed
        // datasets.
        #[test]
        fn nprobe_equals_nlist_always_matches_flat(
            vectors in prop::collection::vec(
                prop::collection::vec(-100.0f32..100.0, 3),
                20..40,
            )
        ) {
            let nlist = 4;
            let mut ivf = IvfIndex::new(3, Metric::L2, nlist, vectors.len(), nlist);
            let mut flat = FlatIndex::new(3, Metric::L2);
            for (id, vector) in vectors.iter().enumerate() {
                ivf.upsert(Record::new(id as u64, vector.clone()));
                flat.upsert(Record::new(id as u64, vector.clone()));
            }
            prop_assert!(ivf.is_trained());
            prop_assert_eq!(ivf.len(), flat.len());

            for query in &vectors {
                let ivf_ids: Vec<u64> = ivf.search(query, 5).iter().map(|h| h.id).collect();
                let flat_ids: Vec<u64> = flat.search(query, 5).iter().map(|h| h.id).collect();
                prop_assert_eq!(ivf_ids, flat_ids);
            }
        }
    }
}