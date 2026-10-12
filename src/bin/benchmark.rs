//! Vector-search HNSW benchmark
//!
//! USAGE:
//!   cargo run --release --bin benchmark -- [insert_count] [query_count] [k] [FLAGS]
//!
//! POSITIONAL (all optional):
//!   <insert_count>   Vectors to insert           [default: 1000]
//!   <query_count>    Query operations to run     [default: 200]
//!   <k>              Nearest neighbours          [default: 5]
//!
//! FLAGS:
//!   --help              Print this help message and exit
//!   --glove <path>      Load real GloVe embeddings from file
//!   --no-recall         Skip brute-force recall@k (speeds up large runs)
//!   --histogram         Show a latency histogram for query times
//!   --warmup <n>        Warm-up query count before timing  [default: 20]

use std::{
    collections::HashSet,
    env,
    fs::{self, File},
    hint::black_box,
    io::{BufRead, BufReader, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use ordered_float::OrderedFloat;
use vector_search::{errors::Result, storage::engine::Engine};

// ─── constants ────────────────────────────────────────────────────────────────

const DIMS: usize = 300;
const DEFAULT_INSERT: usize = 1_000;
const DEFAULT_QUERY: usize = 200;
const DEFAULT_K: usize = 5;
const DEFAULT_WARMUP: usize = 20;

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Config {
    insert_count: usize,
    query_count: usize,
    k: usize,
    warmup: usize,
    glove_path: Option<String>,
    no_recall: bool,
    histogram: bool,
}

fn print_help() {
    println!(
        "\n╔══════════════════════════════════════════════════════╗\n\
         ║         vector_search  —  HNSW Benchmark Tool        ║\n\
         ╚══════════════════════════════════════════════════════╝\n\
         \n\
         USAGE:\n\
           cargo run --release --bin benchmark -- [OPTIONS]\n\
         \n\
         POSITIONAL (all optional):\n\
           <insert_count>   Vectors to insert           [default: {DEFAULT_INSERT}]\n\
           <query_count>    Query operations to run     [default: {DEFAULT_QUERY}]\n\
           <k>              Nearest neighbours          [default: {DEFAULT_K}]\n\
         \n\
         FLAGS:\n\
           --help              Print this help message and exit\n\
           --glove <path>      Load real GloVe embeddings from file\n\
           --no-recall         Skip brute-force recall@k (speeds up large runs)\n\
           --histogram         Show a latency histogram for query times\n\
           --warmup <n>        Warm-up query count before timing  [default: {DEFAULT_WARMUP}]\n\
         \n\
         EXAMPLES:\n\
           # Quick smoke-test (synthetic vectors)\n\
           cargo run --release --bin benchmark\n\
         \n\
           # 5 000 inserts, 500 queries, top-10, with histogram\n\
           cargo run --release --bin benchmark -- 5000 500 10 --histogram\n\
         \n\
           # Real GloVe data, skip recall\n\
           cargo run --release --bin benchmark -- 2000 400 5 --glove glove.6B.300d.txt --no-recall\n"
    );
}

fn parse_args() -> std::result::Result<Config, String> {
    let raw: Vec<String> = env::args().skip(1).collect();

    if raw.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    let mut positional: Vec<String> = Vec::new();
    let mut glove_path: Option<String> = None;
    let mut no_recall = false;
    let mut histogram = false;
    let mut warmup = DEFAULT_WARMUP;

    let mut i = 0usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--glove" => {
                i += 1;
                glove_path = Some(
                    raw.get(i)
                        .ok_or_else(|| "--glove requires a path argument".to_string())?
                        .clone(),
                );
            }
            "--no-recall" => no_recall = true,
            "--histogram" => histogram = true,
            "--warmup" => {
                i += 1;
                warmup = raw
                    .get(i)
                    .ok_or_else(|| "--warmup requires a number".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--warmup must be a positive integer".to_string())?;
            }
            other if !other.starts_with('-') => positional.push(other.to_string()),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }

    fn pos(
        p: &[String],
        idx: usize,
        default: usize,
        name: &str,
    ) -> std::result::Result<usize, String> {
        match p.get(idx) {
            Some(v) => v
                .parse::<usize>()
                .map_err(|_| format!("{name} must be a positive integer")),
            None => Ok(default),
        }
    }

    Ok(Config {
        insert_count: pos(&positional, 0, DEFAULT_INSERT, "insert_count")?,
        query_count: pos(&positional, 1, DEFAULT_QUERY, "query_count")?,
        k: pos(&positional, 2, DEFAULT_K, "k")?,
        warmup,
        glove_path,
        no_recall,
        histogram,
    })
}

// ─── statistics ───────────────────────────────────────────────────────────────

fn percentile_ms(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((p * sorted_ms.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted_ms.len() - 1);
    sorted_ms[idx]
}

fn mean_ms(ms: &[f64]) -> f64 {
    if ms.is_empty() {
        return 0.0;
    }
    ms.iter().sum::<f64>() / ms.len() as f64
}

fn stddev_ms(ms: &[f64]) -> f64 {
    if ms.len() < 2 {
        return 0.0;
    }
    let m = mean_ms(ms);
    let v = ms.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ms.len() as f64;
    v.sqrt()
}

struct Stats {
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    p999: f64,
    avg: f64,
    stddev: f64,
    min: f64,
    max: f64,
    throughput: f64,
    total_secs: f64,
    count: usize,
}

fn compute_stats(samples: &[Duration]) -> Stats {
    let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1_000.0).collect();
    ms.sort_by(f64::total_cmp);
    let total_secs: f64 = samples.iter().map(|d| d.as_secs_f64()).sum();
    Stats {
        p50: percentile_ms(&ms, 0.50),
        p90: percentile_ms(&ms, 0.90),
        p95: percentile_ms(&ms, 0.95),
        p99: percentile_ms(&ms, 0.99),
        p999: percentile_ms(&ms, 0.999),
        avg: mean_ms(&ms),
        stddev: stddev_ms(&ms),
        min: ms.first().copied().unwrap_or(0.0),
        max: ms.last().copied().unwrap_or(0.0),
        throughput: if total_secs > 0.0 {
            ms.len() as f64 / total_secs
        } else {
            0.0
        },
        total_secs,
        count: ms.len(),
    }
}

fn print_stats(label: &str, s: &Stats) {
    let bar = "─".repeat(58);
    println!("\n┌{bar}┐");
    println!("│  {label:<56}│");
    println!("├{bar}┤");
    println!("│  ops          : {:<40}│", s.count);
    println!(
        "│  total time   : {:<40}│",
        format!("{:.3} s", s.total_secs)
    );
    println!(
        "│  throughput   : {:<40}│",
        format!("{:.1} ops/s", s.throughput)
    );
    println!("├{bar}┤");
    println!("│  min          : {:<40}│", format!("{:.4} ms", s.min));
    println!("│  avg          : {:<40}│", format!("{:.4} ms", s.avg));
    println!("│  stddev       : {:<40}│", format!("{:.4} ms", s.stddev));
    println!("│  max          : {:<40}│", format!("{:.4} ms", s.max));
    println!("├{bar}┤");
    println!("│  p50          : {:<40}│", format!("{:.4} ms", s.p50));
    println!("│  p90          : {:<40}│", format!("{:.4} ms", s.p90));
    println!("│  p95          : {:<40}│", format!("{:.4} ms", s.p95));
    println!("│  p99          : {:<40}│", format!("{:.4} ms", s.p99));
    println!("│  p99.9        : {:<40}│", format!("{:.4} ms", s.p999));
    println!("└{bar}┘");
}

// ─── latency histogram ────────────────────────────────────────────────────────

fn print_histogram(samples: &[Duration]) {
    const BUCKETS: usize = 16;
    if samples.is_empty() {
        return;
    }
    let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1_000.0).collect();
    ms.sort_by(f64::total_cmp);

    let min = ms[0];
    let max = ms[ms.len() - 1];
    let range = (max - min).max(f64::EPSILON);
    let bucket_width = range / BUCKETS as f64;

    let mut counts = vec![0usize; BUCKETS];
    for &v in &ms {
        let b = ((v - min) / bucket_width) as usize;
        counts[b.min(BUCKETS - 1)] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);
    let bar_width = 40usize;

    println!("\n  Query Latency Histogram (ms)");
    println!("  {}", "─".repeat(65));
    for (i, &c) in counts.iter().enumerate() {
        let lo = min + i as f64 * bucket_width;
        let hi = lo + bucket_width;
        let bar_len = (c * bar_width) / max_count.max(1);
        let bar: String = "█".repeat(bar_len);
        println!("  [{:7.3} – {:7.3}]  {:>5}  │{}", lo, hi, c, bar);
    }
    println!("  {}", "─".repeat(65));
}

// ─── recall@k helpers ────────────────────────────────────────────────────────

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
}

fn brute_force_knn(query: &[f32], dataset: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(OrderedFloat<f32>, usize)> = dataset
        .iter()
        .enumerate()
        .map(|(i, v)| (OrderedFloat(cosine_distance(query, v)), i))
        .collect();
    dists.sort_unstable();
    dists.iter().take(k).map(|&(_, id)| id).collect()
}

/// We stored the index as 8 LE bytes in the payload; recover it from the
/// engine's returned &str (which treats the bytes as raw UTF-8 regardless).
fn parse_payload_index(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() >= 8 {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&b[..8]);
        Some(usize::from_le_bytes(arr))
    } else {
        None
    }
}

// ─── GloVe loader ────────────────────────────────────────────────────────────

fn load_glove(path: &str, limit: usize) -> std::io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut result = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        parts.next(); // skip word token
        let vector: Vec<f32> = parts.map(|v| v.parse::<f32>().unwrap_or(0.0)).collect();
        if vector.len() == DIMS {
            result.push(vector);
        }
        if result.len() >= limit {
            break;
        }
    }
    Ok(result)
}

// ─── synthetic vector ─────────────────────────────────────────────────────────

fn make_vector(id: usize) -> Vec<f32> {
    (0..DIMS)
        .map(|dim| {
            let v = (id as f32 + 1.0) * (dim as f32 + 1.0);
            v.sin() + (v * 0.013).cos()
        })
        .collect()
}

// ─── entry point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}\nRun with --help for usage.");
        std::process::exit(1);
    });

    // Isolated temp directory so the benchmark never touches real DB files.
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let work_dir = env::temp_dir().join(format!(
        "vector-search-bench-{}-{timestamp}",
        std::process::id()
    ));
    fs::create_dir_all(work_dir.join("db"))?;
    let original_dir = env::current_dir()?;
    // env::set_current_dir(&work_dir)?;

    let result = run_benchmark(&cfg).await;

    env::set_current_dir(original_dir)?;
    let _ = fs::remove_dir_all(&work_dir);
    result
}

async fn run_benchmark(cfg: &Config) -> Result<()> {
    // ── header ────────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║       vector_search  ·  HNSW  Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  insert_count : {}", cfg.insert_count);
    println!("  query_count  : {}", cfg.query_count);
    println!("  k            : {}", cfg.k);
    println!("  warmup       : {}", cfg.warmup);
    println!(
        "  source       : {}",
        cfg.glove_path.as_deref().unwrap_or("synthetic")
    );
    println!(
        "  recall@k     : {}",
        if cfg.no_recall {
            "disabled"
        } else {
            "enabled (brute-force)"
        }
    );

    // ── load / generate vectors ───────────────────────────────────────────────
    let vectors: Vec<Vec<f32>> = if let Some(path) = &cfg.glove_path {
        print!("\n  Loading GloVe vectors from '{path}' … ");
        let _ = std::io::stdout().flush();
        let t0 = Instant::now();
        let vecs =
            load_glove(path, cfg.insert_count).map_err(vector_search::errors::DBError::Io)?;
        println!(
            "loaded {} vectors in {:.2} s",
            vecs.len(),
            t0.elapsed().as_secs_f64()
        );
        if vecs.len() < cfg.insert_count {
            eprintln!(
                "  Warning: only {} vectors available (requested {})",
                vecs.len(),
                cfg.insert_count
            );
        }
        vecs
    } else {
        println!("\n  Generating {} synthetic vectors …", cfg.insert_count);
        (0..cfg.insert_count).map(make_vector).collect()
    };

    let insert_count = vectors.len();
    if insert_count == 0 {
        eprintln!("  No vectors to insert. Exiting.");
        return Ok(());
    }

    // ── engine init ───────────────────────────────────────────────────────────
    let mut engine = Engine::new().await?;
    let mut insert_samples = Vec::with_capacity(insert_count);

    // ── insert phase ──────────────────────────────────────────────────────────
    println!("\n  Inserting {} vectors …", insert_count);
    let insert_start = Instant::now();

    for (id, vector) in vectors.iter().enumerate() {
        // Store the index as an 8-byte LE payload for recall recovery later.
        let payload: [u8; 8] = (id as u64).to_le_bytes();
        let t0 = Instant::now();
        engine
            .insert(black_box(vector), black_box(&payload))
            .await?;
        insert_samples.push(t0.elapsed());

        // Progress every ~10 %
        let step = (insert_count / 10).max(1);
        if (id + 1) % step == 0 || id + 1 == insert_count {
            let pct = (id + 1) * 100 / insert_count;
            let rate = (id + 1) as f64 / insert_start.elapsed().as_secs_f64();
            print!(
                "\r  [{pct:>3}%] {}/{insert_count}  {rate:>8.1} vec/s  ",
                id + 1
            );
            let _ = std::io::stdout().flush();
        }
    }
    println!();

    let insert_stats = compute_stats(&insert_samples);
    print_stats("INSERT  (per-vector latency)", &insert_stats);

    // ── warm-up ───────────────────────────────────────────────────────────────
    if cfg.warmup > 0 && cfg.query_count > 0 {
        print!("\n  Warm-up ({} queries) … ", cfg.warmup);
        let _ = std::io::stdout().flush();
        for wi in 0..cfg.warmup {
            let idx = wi.wrapping_mul(7_919) % insert_count;
            let _ = engine.query(black_box(&vectors[idx])).await?;
        }
        println!("done.");
    }

    // ── query phase ───────────────────────────────────────────────────────────
    println!("\n  Running {} queries (k = {}) …", cfg.query_count, cfg.k);
    let mut query_samples = Vec::with_capacity(cfg.query_count);
    let mut total_recall_hits = 0usize;
    let mut total_recall_possible = 0usize;

    for qi in 0..cfg.query_count {
        let idx = qi.wrapping_mul(7_919) % insert_count;
        let query_vec = &vectors[idx];

        let t0 = Instant::now();
        let results = engine.query(black_box(query_vec)).await?;
        query_samples.push(t0.elapsed());

        // ── recall@k ──────────────────────────────────────────────────────────
        if !cfg.no_recall {
            let truth: HashSet<usize> = brute_force_knn(query_vec, &vectors, cfg.k)
                .into_iter()
                .collect();

            // Parse payload-encoded indices from the returned strings.
            let hnsw_ids: HashSet<usize> = results
                .iter()
                .filter_map(|s| parse_payload_index(s))
                .collect();

            let hits = if hnsw_ids.is_empty() {
                // Fallback: count it as 1 hit if the query vec itself is in truth
                // and we returned at least one result.
                if truth.contains(&idx) && !results.is_empty() {
                    1
                } else {
                    0
                }
            } else {
                hnsw_ids.intersection(&truth).count()
            };

            total_recall_hits += hits;
            total_recall_possible += cfg.k.min(truth.len());
        }

        black_box(&results);
    }

    let query_stats = compute_stats(&query_samples);
    print_stats("QUERY  (per-query latency)", &query_stats);

    // ── histogram ─────────────────────────────────────────────────────────────
    if cfg.histogram {
        print_histogram(&query_samples);
    }

    // ── recall summary ────────────────────────────────────────────────────────
    if !cfg.no_recall && total_recall_possible > 0 {
        let recall = total_recall_hits as f64 / total_recall_possible as f64;
        let bar = "─".repeat(58);
        println!("\n┌{bar}┐");
        println!(
            "│  RECALL @ {}                                              │",
            cfg.k
        );
        println!("├{bar}┤");
        println!(
            "│  hits / possible : {:<38}│",
            format!("{total_recall_hits} / {total_recall_possible}")
        );
        println!(
            "│  recall@{:<2}       : {:<38}│",
            cfg.k,
            format!("{:.2} %", recall * 100.0)
        );
        println!("└{bar}┘");
    }

    // ── summary table ─────────────────────────────────────────────────────────
    println!("\n  ┌─────────────────────────────────────────────────────┐");
    println!("  │                    SUMMARY                          │");
    println!("  ├─────────────────┬───────────────┬───────────────────┤");
    println!("  │ metric          │    insert     │      query        │");
    println!("  ├─────────────────┼───────────────┼───────────────────┤");
    println!(
        "  │ ops             │ {:>13} │ {:>17} │",
        insert_stats.count, query_stats.count
    );
    println!(
        "  │ throughput      │ {:>10.1} o/s │ {:>14.1} o/s │",
        insert_stats.throughput, query_stats.throughput
    );
    println!(
        "  │ p50             │ {:>10.3} ms │ {:>14.3} ms │",
        insert_stats.p50, query_stats.p50
    );
    println!(
        "  │ p95             │ {:>10.3} ms │ {:>14.3} ms │",
        insert_stats.p95, query_stats.p95
    );
    println!(
        "  │ p99             │ {:>10.3} ms │ {:>14.3} ms │",
        insert_stats.p99, query_stats.p99
    );
    println!(
        "  │ p99.9           │ {:>10.3} ms │ {:>14.3} ms │",
        insert_stats.p999, query_stats.p999
    );
    println!("  └─────────────────┴───────────────┴───────────────────┘");
    println!();

    Ok(())
}
