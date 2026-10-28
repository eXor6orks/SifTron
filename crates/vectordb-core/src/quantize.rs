pub struct ScalarQuantizer { mins: Vec<f32>, scales: Vec<f32> }
use crate::distance::{Distance, Metric};

impl ScalarQuantizer {
    pub fn train(vectors: &[Vec<f32>]) -> Self {
        assert!(!vectors.is_empty(), "cannot train a quantizer on zero vectors");
        let dimension = vectors[0].len();
        let mut mins = vec![f32::MAX; vectors[0].len()];
        let mut maxs = vec![f32::MIN; vectors[0].len()];
        for vector in vectors {
            assert_eq!(vector.len(), dimension, "all training vectors must have the same dimension");
            for (i, &v) in vector.iter().enumerate() {
                if v < mins[i] {
                    mins[i] = v;
                }
                if v > maxs[i] {
                    maxs[i] = v;
                }
            }
        }
        let scales: Vec<f32> = maxs
            .iter()
            .zip(mins.iter())
            .map(|(&max, &min)| {
                let range = max - min;
                // A constant dimension has range 0. A scale of 0 would make `encode`
                // divide by zero (0/0 = NaN), so use any non-zero value: in-range
                // values still encode to code 0 exactly.
                if range > 0.0 { range / 255.0 } else { 1.0 }
            })
            .collect();
        ScalarQuantizer { mins, scales }
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.mins.len(), "vector dimension does not match the quantizer");
        let mut codes = Vec::with_capacity(vector.len());
        for(i, &v) in vector.iter().enumerate() {
            let code = ((v - self.mins[i]) / self.scales[i]).round().clamp(0.0, 255.0) as u8;
            codes.push(code);
        }
        codes
    }

    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(code.len(), self.mins.len(), "code dimension does not match the quantizer");
        let mut vectors = Vec::with_capacity(code.len());
        for (i, &c) in code.iter().enumerate() {
            vectors.push(self.mins[i] + (c as f32) * self.scales[i]);
        }
        vectors
    }

    pub fn distance(&self, metric: &Metric, query: &[f32], code: &[u8]) -> f32 {
        match metric {
            Metric::L2 => {
                // Hot path: this runs once per stored vector, for every
                // query. Decoding into a fresh `Vec` here meant one heap
                // allocation per comparison, so for L2 we compute the
                // distance directly from the codes instead.
                assert_eq!(query.len(), self.mins.len(), "query dimension does not match the quantizer");
                assert_eq!(code.len(), self.mins.len(), "code dimension does not match the quantizer");
                let mut sum = 0.0f32;
                for i in 0..code.len() {
                    // same arithmetic as `decode` followed by `L2::distance`,
                    // in the same order, so the result is identical
                    let decoded = self.mins[i] + code[i] as f32 * self.scales[i];
                    let diff = query[i] - decoded;
                    sum += diff * diff;
                }
                sum
            }
            // Cosine and Dot keep the generic path (decode, then the
            // project's own metric): same edge-case behaviour as FlatIndex,
            // e.g. Cosine of a zero vector.
            _ => metric.distance(query, &self.decode(code)),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // dim 0: 0..255 (scale exactly 1.0) | dim 1: -10..10 | dim 2: constant 5.0
    fn quantizer() -> ScalarQuantizer {
        ScalarQuantizer::train(&[
            vec![0.0, -10.0, 5.0],
            vec![255.0, 10.0, 5.0],
            vec![100.0, 0.0, 5.0],
        ])
    }

    #[test]
    fn extremes_map_to_codes_0_and_255() {
        let q = quantizer();
        assert_eq!(q.encode(&[0.0, -10.0, 5.0]), vec![0, 0, 0]);
        assert_eq!(q.encode(&[255.0, 10.0, 5.0]), vec![255, 255, 0]);
    }

    #[test]
    fn round_trip_error_is_at_most_half_a_step() {
        let q = quantizer();
        let v = [37.3f32, 3.3, 5.0];
        let back = q.decode(&q.encode(&v));
        for i in 0..3 {
            assert!(
                (back[i] - v[i]).abs() <= q.scales[i] / 2.0 + 1e-4,
                "dim {i}: {} vs {}", back[i], v[i]
            );
        }
    }

    #[test]
    fn re_encoding_a_decoded_vector_gives_the_same_code() {
        let q = quantizer();
        let code = q.encode(&[37.3, 3.3, 5.0]);
        assert_eq!(q.encode(&q.decode(&code)), code);
    }

    #[test]
    fn constant_dimension_never_produces_nan() {
        let q = quantizer();
        assert_eq!(q.decode(&q.encode(&[10.0, 0.0, 5.0]))[2], 5.0);
        // a value outside the zero-width learned range must stay finite
        let back = q.decode(&q.encode(&[10.0, 0.0, 9.0]));
        assert!(back.iter().all(|x| x.is_finite()), "got {back:?}");
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let q = quantizer();
        assert_eq!(q.encode(&[300.0, 0.0, 5.0])[0], 255);
        assert_eq!(q.encode(&[-40.0, 0.0, 5.0])[0], 0);
    }

    #[test]
    fn distance_equals_the_metric_applied_to_the_decoded_vector() {
        let q = quantizer();
        let query = [10.0, 1.0, 5.0];
        let code = q.encode(&[50.0, -2.0, 5.0]);
        let decoded = q.decode(&code);
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            assert_eq!(
                q.distance(&metric, &query, &code),
                metric.distance(&query, &decoded),
                "{metric:?}"
            );
        }
    }

    #[test]
    fn cosine_with_a_zero_query_is_defined() {
        let q = quantizer();
        let code = q.encode(&[50.0, -2.0, 5.0]);
        assert_eq!(q.distance(&Metric::Cosine, &[0.0, 0.0, 0.0], &code), 1.0);
    }

    #[test]
    #[should_panic(expected = "dimension")]
    fn encode_rejects_a_shorter_vector() {
        quantizer().encode(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "dimension")]
    fn encode_rejects_a_longer_vector() {
        quantizer().encode(&[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "zero vectors")]
    fn train_rejects_empty_input() {
        ScalarQuantizer::train(&[]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        // Whatever the training data, a vector from that data comes back
        // within half a quantization step on every dimension.
        #[test]
        fn round_trip_error_bound_holds_for_random_data(
            vectors in prop::collection::vec(prop::collection::vec(-100.0f32..100.0, 4), 2..30)
        ) {
            let q = ScalarQuantizer::train(&vectors);
            for v in &vectors {
                let back = q.decode(&q.encode(v));
                for i in 0..4 {
                    prop_assert!((back[i] - v[i]).abs() <= q.scales[i] / 2.0 + 1e-3);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        // The optimised L2 path must give exactly what the generic path
        // (decode, then the project's metric) gives, whatever the data.
        #[test]
        fn direct_l2_matches_decode_then_metric(
            vectors in prop::collection::vec(prop::collection::vec(-100.0f32..100.0, 5), 2..20),
            query in prop::collection::vec(-150.0f32..150.0, 5),
        ) {
            let q = ScalarQuantizer::train(&vectors);
            for v in &vectors {
                let code = q.encode(v);
                let direct = q.distance(&Metric::L2, &query, &code);
                let generic = Metric::L2.distance(&query, &q.decode(&code));
                prop_assert_eq!(direct, generic);
            }
        }
    }
}