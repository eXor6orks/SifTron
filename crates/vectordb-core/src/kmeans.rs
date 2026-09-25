use rand::seq::index::sample;
use rand::Rng;

use crate::distance::{Distance, Metric};

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

pub fn kmeans(
    vectors: &[Vec<f32>],
    k: usize,
    metric: &Metric,
    max_iterations: usize,
    rng: &mut impl Rng,
) -> KMeansResult {
    assert!(k > 0, "k must be at least 1");
    assert!(
        vectors.len() >= k,
        "need at least as many vectors ({}) as clusters ({k})",
        vectors.len()
    );

    let dimension = vectors[0].len();

    let mut centroids: Vec<Vec<f32>> = sample(rng, vectors.len(), k)
        .iter()
        .map(|i| vectors[i].clone())
        .collect();

    let mut assignments = vec![0usize; vectors.len()];
    let mut previous_inertia = f32::INFINITY;

    for _ in 0..max_iterations {
        let mut inertia = 0.0f32;
        for (i, vector) in vectors.iter().enumerate() {
            let (closest, distance) = nearest(vector, &centroids, metric);
            assignments[i] = closest;
            inertia += distance;
        }

        let mut sums = vec![vec![0.0f32; dimension]; k];
        let mut counts = vec![0usize; k];
        for (vector, &cluster) in vectors.iter().zip(&assignments) {
            counts[cluster] += 1;
            for (sum_component, &value) in sums[cluster].iter_mut().zip(vector) {
                *sum_component += value;
            }
        }

        for j in 0..k {
            if counts[j] == 0 {
                // Empty cluster: steal the vector currently farthest from
                // its own centroid, so no centroid is ever averaged over
                // zero points — which would otherwise divide by zero and
                // silently produce NaN.
                let (worst_index, _) = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, metric.distance(v, &centroids[assignments[i]])))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("vectors is non-empty (checked above)");
                centroids[j] = vectors[worst_index].clone();
                assignments[worst_index] = j;
            } else {
                centroids[j] = sums[j].iter().map(|&sum| sum / counts[j] as f32).collect();
            }
        }

        if previous_inertia - inertia < 1e-6 {
            break;
        }
        previous_inertia = inertia;
    }

    KMeansResult {
        centroids,
        assignments,
    }
}

/// Returns the index of the centroid closest to `vector`, and the distance
/// to it.
fn nearest(vector: &[f32], centroids: &[Vec<f32>], metric: &Metric) -> (usize, f32) {
    centroids
        .iter()
        .enumerate()
        .map(|(j, c)| (j, metric.distance(vector, c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("centroids is non-empty (k > 0, checked by caller)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn separates_two_well_separated_clusters() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![-0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![9.9, 10.0],
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let result = kmeans(&vectors, 2, &Metric::L2, 25, &mut rng);

        let group_a = result.assignments[0];
        let group_b = result.assignments[4];
        assert_ne!(
            group_a, group_b,
            "the two blobs must land in different clusters"
        );
        assert!(result.assignments[0..4].iter().all(|&a| a == group_a));
        assert!(result.assignments[4..8].iter().all(|&a| a == group_b));
    }

    #[test]
    fn k_equals_len_gives_zero_inertia() {
        let vectors = vec![vec![0.0], vec![5.0], vec![10.0]];
        let mut rng = StdRng::seed_from_u64(1);
        let result = kmeans(&vectors, 3, &Metric::L2, 10, &mut rng);
        for v in &vectors {
            assert!(result.centroids.iter().any(|c| (c[0] - v[0]).abs() < 1e-6));
        }
    }

    #[test]
    fn duplicate_vectors_never_produce_nan_centroids() {
        let vectors = vec![vec![0.0, 0.0]; 4];
        let mut rng = StdRng::seed_from_u64(7);
        let result = kmeans(&vectors, 4, &Metric::L2, 10, &mut rng);
        for centroid in &result.centroids {
            for &x in centroid {
                assert!(x.is_finite());
            }
        }
    }

    #[test]
    fn same_seed_gives_same_result() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![8.0, 8.0],
            vec![9.0, 9.0],
        ];
        let mut rng_a = StdRng::seed_from_u64(99);
        let mut rng_b = StdRng::seed_from_u64(99);
        let a = kmeans(&vectors, 2, &Metric::L2, 25, &mut rng_a);
        let b = kmeans(&vectors, 2, &Metric::L2, 25, &mut rng_b);
        assert_eq!(a.assignments, b.assignments);
    }

    #[test]
    #[should_panic]
    fn panics_when_k_is_zero() {
        let vectors = vec![vec![0.0], vec![1.0]];
        let mut rng = StdRng::seed_from_u64(0);
        kmeans(&vectors, 0, &Metric::L2, 10, &mut rng);
    }

    #[test]
    #[should_panic]
    fn panics_when_k_exceeds_vector_count() {
        let vectors = vec![vec![0.0], vec![1.0]];
        let mut rng = StdRng::seed_from_u64(0);
        kmeans(&vectors, 5, &Metric::L2, 10, &mut rng);
    }
}
