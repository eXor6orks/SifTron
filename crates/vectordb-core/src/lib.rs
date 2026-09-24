//! Core traits and the exact baseline index for the SifTron vector
//! database. See `IndexStrategy` for the interface every index backend
//! (`FlatIndex` today, `IvfIndex`/`HnswIndex` in later versions)
//! implements.

mod distance;
mod flat;
mod index;
mod record;

pub use distance::{Cosine, Distance, DotProduct, Metric, L2};
pub use flat::FlatIndex;
pub use index::IndexStrategy;
pub use record::{Hit, Record};
