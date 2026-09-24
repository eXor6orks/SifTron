# SifTron

A vector search engine built from scratch in Rust — usable from its first
version, gaining capabilities (approximate indexes, quantization,
persistence, combined metadata filtering) release by release rather than
being designed all at once.

## Status: v0.1.0 — exact baseline

- `Distance` trait with `L2`, `Cosine`, `DotProduct` implementations.
- `IndexStrategy` trait — the interface every index backend implements.
- `FlatIndex`: exhaustive brute-force search. This is the ground truth
  (`recall@k = 1.0` by construction) that later approximate indexes (IVF,
  HNSW) are measured against, and it stays available as a selectable
  backend afterwards — a small collection can stay exact with zero tuning.

## Building

```sh
cargo build --workspace
cargo test --workspace
```

MSRV: 1.96.0 (pinned in CI; `cargo build` with an older toolchain is not
supported).

## Layout

| Crate | Role |
| --- | --- |
| `vectordb-core` | `Distance`, `IndexStrategy`, `Record`/`Hit`, `FlatIndex` |

Future crates (`vectordb-quantize`, `vectordb-storage`, `vectordb-filter`,
`vectordb-server`) land as the corresponding milestones below are built.

## Roadmap

| Version | Milestone |
| --- | --- |
| v0.1.0 | Exact baseline: `Distance`, `FlatIndex` *(this release)* |
| v0.2.0 | IVF (inverted file index) |
| v0.3.0 | HNSW (hierarchical navigable small world) |
| v0.4.0 | Quantization (SQ8, PQ) |
| v0.5.0 | Persistence: mmap segments, WAL, recovery |
| v0.6.0 | Combined metadata filtering |
| v0.7.0 | gRPC/REST API, configuration, CLI |
| v0.8.0 | Concurrency & robustness |
| v0.9.0 | Observability |
| v1.0.0 | Docker image, comparative benchmarks, release |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
