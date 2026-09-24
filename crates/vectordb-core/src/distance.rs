//! Distance metrics between vectors.

/// A metric that scores similarity or dissimilarity between two vectors of
/// equal dimension. Implementations must be deterministic and symmetric
/// unless documented otherwise.
pub trait Distance {
    /// Computes the distance between `a` and `b`. Lower is closer.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Squared Euclidean (L2) distance. Monotonic with true L2 distance and
/// avoids an unnecessary `sqrt`, which is why nearest-neighbor search uses
/// it directly instead of the true L2 norm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L2;

impl Distance for L2 {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}

/// Cosine distance, defined as `1 - cosine_similarity(a, b)`, so that a
/// smaller value means more similar (consistent with other `Distance`
/// implementations, where lower is closer).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cosine;

impl Distance for Cosine {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|y| y * y).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        1.0 - dot / (norm_a * norm_b)
    }
}

/// Negative dot product, so that lower still means closer/more similar,
/// consistent with [`L2`] and [`Cosine`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DotProduct;

impl Distance for DotProduct {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
        -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }
}

/// The metric selected for a collection, as configured by the user (see
/// `metric = "cosine" | "l2" | "dot"` in the collection schema).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
    Dot,
}

impl Distance for Metric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => L2.distance(a, b),
            Metric::Cosine => Cosine.distance(a, b),
            Metric::Dot => DotProduct.distance(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_identical_vectors_is_zero() {
        let v = [1.0, 2.0, 3.0];
        assert_eq!(L2.distance(&v, &v), 0.0);
    }

    #[test]
    fn l2_matches_known_value() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_eq!(L2.distance(&a, &b), 25.0); // 3^2 + 4^2
    }

    #[test]
    fn cosine_identical_direction_is_zero() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0]; // same direction, different magnitude
        assert!(Cosine.distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((Cosine.distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_defined() {
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];
        assert_eq!(Cosine.distance(&a, &b), 1.0);
    }

    #[test]
    fn dot_product_prefers_larger_projection() {
        let query = [1.0, 0.0];
        let close = [2.0, 0.0];
        let far = [0.5, 0.0];
        assert!(DotProduct.distance(&query, &close) < DotProduct.distance(&query, &far));
    }

    #[test]
    #[should_panic]
    fn l2_panics_on_dimension_mismatch() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        L2.distance(&a, &b);
    }
}
