//! Interactive REPL for hand-testing `FlatIndex` without writing Rust.
//! This is a throwaway playground, not the future `vectordb-server` (v0.7.0)
//! API/CLI described in the engineering plan.
//!
//! Run with: `cargo run -- <dimension> <metric>`
//! `<metric>` is one of `l2`, `cosine`, `dot` (default: `cosine`).

use std::io::{self, BufRead, Write};

use vectordb_core::{FlatIndex, IndexStrategy, Metric, Record};

fn parse_metric(s: &str) -> Option<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "l2" => Some(Metric::L2),
        "cosine" => Some(Metric::Cosine),
        "dot" => Some(Metric::Dot),
        _ => None,
    }
}

fn parse_vector(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .map_err(|e| format!("invalid number '{}': {e}", part.trim()))
        })
        .collect()
}

fn print_help() {
    println!("commands:");
    println!("  add <id> <v1,v2,...>       insert or overwrite a vector");
    println!("  remove <id>                delete a vector");
    println!("  search <k> <v1,v2,...>     find the k nearest vectors");
    println!("  len                        number of vectors stored");
    println!("  help                       show this message");
    println!("  quit                       exit");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dimension: usize = args
        .next()
        .expect("usage: playground <dimension> [metric]")
        .parse()
        .expect("dimension must be a positive integer");
    let metric = args
        .next()
        .map(|s| parse_metric(&s).unwrap_or_else(|| panic!("unknown metric '{s}'")))
        .unwrap_or(Metric::Cosine);

    let mut index = FlatIndex::new(dimension, metric);
    println!("SifTron playground — dimension={dimension}, metric={metric:?}");
    print_help();

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, ' ');
        let cmd = parts.next().unwrap_or("");

        match cmd {
            "add" => {
                let (Some(id), Some(vector)) = (parts.next(), parts.next()) else {
                    println!("usage: add <id> <v1,v2,...>");
                    continue;
                };
                let id: u64 = match id.parse() {
                    Ok(id) => id,
                    Err(e) => {
                        println!("invalid id: {e}");
                        continue;
                    }
                };
                match parse_vector(vector) {
                    Ok(v) if v.len() != dimension => {
                        println!("expected {dimension} components, got {}", v.len());
                    }
                    Ok(v) => {
                        index.upsert(Record::new(id, v));
                        println!("ok ({} vectors stored)", index.len());
                    }
                    Err(e) => println!("{e}"),
                }
            }
            "remove" => {
                let Some(id) = parts.next() else {
                    println!("usage: remove <id>");
                    continue;
                };
                match id.parse::<u64>() {
                    Ok(id) => println!("removed={}", index.remove(id)),
                    Err(e) => println!("invalid id: {e}"),
                }
            }
            "search" => {
                let (Some(k), Some(vector)) = (parts.next(), parts.next()) else {
                    println!("usage: search <k> <v1,v2,...>");
                    continue;
                };
                let k: usize = match k.parse() {
                    Ok(k) => k,
                    Err(e) => {
                        println!("invalid k: {e}");
                        continue;
                    }
                };
                match parse_vector(vector) {
                    Ok(v) if v.len() != dimension => {
                        println!("expected {dimension} components, got {}", v.len());
                    }
                    Ok(v) => {
                        for hit in index.search(&v, k) {
                            println!("  id={} score={:.4}", hit.id, hit.score);
                        }
                    }
                    Err(e) => println!("{e}"),
                }
            }
            "len" => println!("{}", index.len()),
            "help" => print_help(),
            "quit" | "exit" => break,
            other => println!("unknown command '{other}' (try 'help')"),
        }
    }
}
