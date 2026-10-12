use rustfft::{
    FftPlanner,
    num_complex::{Complex32, Complex64},
};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};
use vector_search::{
    errors::Result,
    storage::engine::{self, Engine},
};

// TODO: Fix vector ID and file offset handling
// TODO: Fix and unify distance implementations
// TODO: Implement graph log replay
// TODO: Restore the engine state on startup
// TODO: Rebuild the payload ID-to-offset index
// TODO: Add an engine-level query API
// TODO: Make HNSW layer capacities consistent
// TODO: Make random level generation restart-safe
// TODO: Add database directory and metadata handling
// TODO: Add crash consistency and recovery handling
// TODO: Add persistence and graph correctness tests

fn load_glove(path: &str, limit: Option<usize>) -> std::io::Result<Vec<(String, Vec<f32>)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut result = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();

        let Some(word) = parts.next() else {
            continue;
        };

        let vector: Vec<f32> = parts
            .map(|value| value.parse::<f32>().expect("invalid float in GloVe file"))
            .collect();

        result.push((word.to_string(), vector));

        if let Some(limit) = limit {
            if result.len() >= limit {
                break;
            }
        }
    }

    Ok(result)
}

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new().await?;
    // engine.restore().await?;

    let embeddings = load_glove(&format!("glove.6B.{}d.txt", 300), Some(20000))?;
    let t0 = Instant::now();

    let (word, v) = embeddings.iter().nth(175).unwrap();

    for (word, vector) in embeddings.iter() {
        let bytes: Vec<u8> = word.bytes().into_iter().collect();
        engine.insert(&vector, &bytes).await.unwrap();
    }

    println!("{:?} secs", t0.elapsed().as_secs());
    let res = engine.query(&v).await?;
    println!("{word} -> {res:?}");
    // let (_d, payload) = engine.get_payload(0).await?;
    // println!("{:?}", String::from_utf8(payload).unwrap());
    Ok(())
}
