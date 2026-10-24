//! Core traits and the exact baseline index for the SifTron vector
//! database. See `IndexStrategy` for the interface every index backend
//! (`FlatIndex` today, `IvfIndex`/`HnswIndex` in later versions)
//! implements.

mod distance;
mod flat;
mod index;
mod kmeans;
mod record;
mod ivf;
mod hnsw;
mod quantize;
mod flat_q8;

pub use distance::{Cosine, Distance, DotProduct, Metric, L2};
pub use flat::FlatIndex;
pub use index::IndexStrategy;
pub use kmeans::{kmeans, KMeansResult};
pub use record::{Hit, Record};
pub use ivf::IvfIndex;
pub use hnsw::HnswIndex;
pub use quantize::ScalarQuantizer;
pub use flat_q8::FlatSq8Index;