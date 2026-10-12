use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::atomic::{
        AtomicU64,
        Ordering::{self},
    },
};

use ordered_float::NotNan;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    errors::Result,
    memgraph::node::Node,
    storage::{engine::VectorProvider, graph_store::GraphLogRecord},
};

pub struct HNSW<P> {
    entry_point_id: Option<u64>,
    max_layer: u64,
    nodes: HashMap<u64, Node>,
    m: usize,               // maximum number of outgoing links/node/layer
    m_max0: usize,          // maximum links allowed at layer 0, 2 * m
    ef_construction: usize, // size of the nearest-neighbors candidate pool during construction
    m_l: f64,               //level probability normalization factor, 1 / ln(m)
    provider: P,
    latest_id: AtomicU64,
    rng: StdRng,
}

impl<P> HNSW<P> {
    pub fn new(provider: P, m: usize, m_max0: usize, ef_construction: usize, m_l: f64) -> Self {
        Self {
            entry_point_id: None,
            max_layer: 0,
            nodes: HashMap::new(),
            m_max0,
            m_l,
            m,
            ef_construction,
            provider,
            latest_id: AtomicU64::new(0),
            rng: rand::rngs::StdRng::seed_from_u64(123),
        }
    }
}

impl<P: VectorProvider> HNSW<P> {
    fn distance(lhs: &[f32], rhs: &[f32]) -> NotNan<f32> {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return NotNan::new(l2_squared_avx2(lhs, rhs)).unwrap();
            }
        }
        assert_eq!(lhs.len(), rhs.len());

        let mut dot = 0.0;
        let mut lhs_norm = 0.0;
        let mut rhs_norm = 0.0;

        for (a, b) in lhs.iter().zip(rhs.iter()) {
            dot += a * b;
            lhs_norm += a * a;
            rhs_norm += b * b;
        }

        let denom = lhs_norm.sqrt() * rhs_norm.sqrt();

        let similarity = if denom == 0.0 { 0.0 } else { dot / denom };

        let distance = 1.0 - similarity;

        NotNan::new(distance).unwrap()
    }
    fn random_level(&mut self) -> u64 {
        let r: f64 = self.rng.random();
        (-r.ln() * self.m_l).floor() as u64
    }
    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub async fn query(&self, vector: &[f32], k: usize) -> Result<Vec<(NotNan<f32>, u64)>> {
        let Some(mut curr_entry) = self.entry_point_id else {
            return Ok(Vec::new());
        };

        // 1. Greedy descent through upper layers
        for layer in (1..=self.max_layer).rev() {
            curr_entry = self.search_layer_closest(vector, curr_entry, layer).await?;
        }

        // 2. Wider search at layer 0
        let mut results = self.search_layer(vector, curr_entry, 0).await?;

        // closest first
        results.truncate(k);

        Ok(results)
    }
    async fn search_layer_closest(&self, vector: &[f32], node_id: u64, layer: u64) -> Result<u64> {
        let node = self.nodes.get(&node_id).unwrap();

        let mut closest_neighbor: u64 = node_id;
        let neighbor_vec = self.provider.get_vector(node.id).await?;
        let mut closest_distance = Self::distance(vector, &neighbor_vec);
        if let Some(neighbors) = node.neighbors.get(&layer) {
            for neighbor in neighbors {
                let neighbor_node_vec = self.provider.get_vector(neighbor.1).await?;
                let distance = Self::distance(vector, &neighbor_node_vec);
                if distance < closest_distance {
                    closest_distance = distance;
                    closest_neighbor = neighbor.1;
                }
            }
        }
        Ok(closest_neighbor)
    }
    async fn search_layer(
        &self,
        vector: &[f32],
        entry: u64,
        layer: u64,
    ) -> Result<Vec<(NotNan<f32>, u64)>> {
        let mut visited = HashSet::new();

        let mut candidates: BTreeSet<(NotNan<f32>, u64)> = BTreeSet::new();
        let mut results: BTreeSet<(NotNan<f32>, u64)> = BTreeSet::new();

        let entry_node = self.nodes.get(&entry).unwrap();
        let entry_node_vec = self.provider.get_vector(entry_node.id).await?;
        let entry_dist = Self::distance(vector, &entry_node_vec);

        candidates.insert((entry_dist, entry));
        results.insert((entry_dist, entry));
        visited.insert(entry);
        while let Some(&(candidate_dist, candidate_id)) = candidates.first() {
            candidates.pop_first();
            let worst_result_dist = results.last().unwrap().0;
            if results.len() >= self.ef_construction && candidate_dist > worst_result_dist {
                break;
            }
            let candidate_node = self.nodes.get(&candidate_id).unwrap();
            if let Some(neighbors) = candidate_node.neighbors.get(&layer) {
                for &neighbor_id in neighbors.iter() {
                    if !visited.insert(neighbor_id.1) {
                        continue;
                    }
                    let neighbor_node_vec = self.provider.get_vector(neighbor_id.1).await?;
                    let dist = Self::distance(vector, &neighbor_node_vec);
                    let good_enough =
                        results.len() < self.ef_construction || dist < results.last().unwrap().0;
                    if good_enough {
                        candidates.insert((dist, neighbor_id.1));
                        results.insert((dist, neighbor_id.1));
                        if results.len() > self.ef_construction {
                            results.pop_last();
                        }
                    }
                }
            }
        }
        let num_results = if layer == 0 { self.m } else { self.m_max0 };
        Ok(results.into_iter().take(num_results).collect())
    }
    pub async fn insert(&mut self, node_id: u64, vector: &[f32]) -> Result<Vec<GraphLogRecord>> {
        // TODO: check whether increase of id should be after insertions
        self.latest_id.fetch_add(1, Ordering::Relaxed);
        let level = self.random_level();
        let mut new_node = Node {
            id: node_id,
            neighbors: HashMap::new(),
        };
        let mut records = vec![GraphLogRecord::NodeAdded { id: node_id }];

        // first node insertion
        if self.entry_point_id.is_none() {
            self.entry_point_id = Some(node_id);
            self.max_layer = level;
            self.nodes.insert(node_id, new_node.clone());
            records.push(GraphLogRecord::EntryPointSet { id: node_id });
            records.push(GraphLogRecord::MaxLayerSet { layer: level });
            return Ok(records);
        }

        let mut curr_entry = self.entry_point_id.unwrap();
        let curr_max_layer = self.max_layer;

        if curr_max_layer > level {
            for layer in ((level + 1)..=curr_max_layer).rev() {
                curr_entry = self.search_layer_closest(vector, curr_entry, layer).await?;
            }
        }
        let start_layer = level.min(curr_max_layer);
        for layer in (0..=start_layer).rev() {
            let neighbors = self.search_layer(vector, curr_entry, layer).await?;
            for &entry in &neighbors {
                let existing_node = self.nodes.get_mut(&entry.1).unwrap();
                let (connect, evicted) =
                    existing_node.add_neighbor(new_node.id, layer, entry.0, self.m);
                if connect {
                    new_node.neighbors.entry(layer).or_default().insert(entry);
                    records.push(GraphLogRecord::InsertEdge {
                        node_id: new_node.id,
                        layer,
                        neighbor_id: entry.1,
                        distance: entry.0.into_inner(),
                    });
                }

                if let Some(evicted_id) = evicted {
                    self.nodes.get_mut(&evicted_id).and_then(|node| {
                        Some(node.neighbors.entry(layer).and_modify(|btree| {
                            let entry = btree.iter().copied().find(|&e| entry.1 == e.1).unwrap();
                            btree.remove(&entry);
                        }))
                    });
                    records.push(GraphLogRecord::DeleteEdge {
                        node_id: entry.1,
                        layer,
                        neighbor_id: evicted_id,
                    });
                }
            }
            // pick a good entry point for the next lower layer
            if let Some(&best) = neighbors.first() {
                curr_entry = best.1;
            }
        }

        self.nodes.insert(node_id, new_node.clone());

        // New highest-level node becomes global entry point.
        if level > curr_max_layer {
            self.max_layer = level;
            self.entry_point_id = Some(node_id);
            records.push(GraphLogRecord::EntryPointSet { id: node_id });
            records.push(GraphLogRecord::MaxLayerSet { layer: level });
        }
        Ok(records)
    }

    pub fn get_lateset_id(&self) -> u64 {
        self.latest_id.load(Ordering::Relaxed)
    }

    pub fn insert_nodes(&mut self, nodes: Vec<Node>) -> u64 {
        for node in nodes {
            let id = node.id;
            self.nodes.insert(id, node);
            self.latest_id.swap(id, Ordering::Relaxed);
        }
        self.latest_id.load(Ordering::Relaxed)
    }

    pub fn set_entry_point(&mut self, entry_point: u64) {
        self.entry_point_id = Some(entry_point);
    }

    pub fn set_max_layer(&mut self, max_layer: u64) {
        self.max_layer = max_layer;
    }

    pub fn print(&self) {
        for node in self.nodes.iter() {
            println!("{} -> {:?}", node.0, node.1)
        }
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    // let a_f32 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };

    // let b_f32 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };

    let mut acc = _mm256_setzero_ps();

    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(offset)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(offset)) };

        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);

        acc = _mm256_add_ps(acc, sq);
    }

    let mut tmp = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };

    let mut result: f32 = tmp.iter().sum();

    for i in chunks * 8..a.len() {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}
