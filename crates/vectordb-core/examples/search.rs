use vectordb_core::{FlatIndex, IndexStrategy, Metric, Record};

fn main() {
    let mut index = FlatIndex::new(3, Metric::Cosine);
    index.upsert(Record::new(1, vec![1.0, 0.0, 0.0]));
    index.upsert(Record::new(2, vec![0.0, 1.0, 0.0]));
    index.upsert(Record::new(3, vec![0.9, 0.1, 0.0]));

    let hits = index.search(&[1.0, 0.0, 0.0], 2);
    for hit in hits {
        println!("id={} score={:.4}", hit.id, hit.score);
    }
}
