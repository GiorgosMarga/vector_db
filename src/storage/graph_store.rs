use std::{
    collections::HashMap,
    io::SeekFrom::Start,
    os::unix::fs::MetadataExt,
    sync::atomic::{AtomicBool, Ordering::Relaxed},
};

use ordered_float::NotNan;
use tokio::{
    fs::{self, File},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, BufWriter},
    sync::Mutex,
};

use crate::{
    errors::{DBError, Result},
    memgraph::node::Node,
};

pub type LayerNeighbors = Vec<(NotNan<f32>, u64)>;
pub struct RestoreResult {
    pub nodes: Vec<Node>,
    pub entry_point: Option<u64>,
    pub max_layer: Option<u64>,
}
const ENTRY_POINT_INDEX: u64 = 0;
const MAX_LAYER_INDEX: u64 = 8;

#[derive(Debug, Clone)]
pub enum GraphLogRecord {
    NodeAdded {
        id: u64,
    },
    InsertEdge {
        node_id: u64,
        layer: u64,
        neighbor_id: u64,
        distance: f32,
    },
    DeleteEdge {
        node_id: u64,
        layer: u64,
        neighbor_id: u64,
    },
    EntryPointSet {
        id: u64,
    },
    MaxLayerSet {
        layer: u64,
    },
}

pub struct GraphStore {
    f: Mutex<BufWriter<File>>,
    dirty: AtomicBool,
}

impl GraphStore {
    pub async fn new() -> Result<GraphStore> {
        let f = Mutex::new(BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open("db/graph.bin")
                .await?,
        ));
        Ok(Self {
            f,
            dirty: AtomicBool::new(false),
        })
    }

    pub async fn flush(&self) -> Result<()> {
        if self.dirty.swap(false, Relaxed) {
            let mut f = self.f.lock().await;
            f.flush().await?; // BufWriter -> kernel
            f.get_ref().sync_all().await?; // kernel -> disk
        }

        Ok(())
    }

    pub async fn insert(&self, records: &[GraphLogRecord]) -> Result<u64> {
        let mut buf = vec![];

        for record in records {
            match record {
                GraphLogRecord::NodeAdded { id } => {
                    buf.write_u8(0).await?;
                    buf.write_u64_le(*id).await?;
                }
                GraphLogRecord::InsertEdge {
                    node_id,
                    layer,
                    neighbor_id,
                    distance,
                } => {
                    buf.write_u8(1).await?;
                    buf.write_u64_le(*node_id).await?;
                    buf.write_u64_le(*layer).await?;
                    buf.write_u64_le(*neighbor_id).await?;
                    buf.write_f32_le(*distance).await?;
                }
                GraphLogRecord::DeleteEdge {
                    node_id,
                    layer,
                    neighbor_id,
                } => {
                    buf.write_u8(2).await?;
                    buf.write_u64_le(*node_id).await?;
                    buf.write_u64_le(*layer).await?;
                    buf.write_u64_le(*neighbor_id).await?;
                }
                GraphLogRecord::EntryPointSet { id } => {
                    buf.write_u8(3).await?;
                    buf.write_u64_le(*id).await?;
                }
                GraphLogRecord::MaxLayerSet { layer } => {
                    buf.write_u8(4).await?;
                    buf.write_u64_le(*layer).await?;
                }
            }
        }
        self.f.lock().await.write_all(&buf).await?;
        self.dirty.store(true, Relaxed);
        Ok(buf.len() as u64)
    }

    pub async fn restore(&self) -> Result<Option<RestoreResult>> {
        let mut nodes = HashMap::new();
        let mut entry_point = None;
        let mut max_layer = None;
        let mut f = self.f.lock().await;
        let size = f.get_ref().metadata().await?.size();
        if size == 0 {
            return Ok(None);
        }
        f.seek(Start(0)).await?;

        let mut f = BufReader::new(&mut *f);
        while let Ok(t) = f.read_u8().await {
            match t {
                0 => {
                    let id = f.read_u64_le().await?;
                    let node = Node {
                        id,
                        neighbors: HashMap::new(),
                    };
                    nodes.insert(id, node);
                }
                1 => {
                    let node_id = f.read_u64_le().await?;
                    let layer = f.read_u64_le().await?;
                    let neighbor_id = f.read_u64_le().await?;
                    let distance = unsafe { NotNan::new_unchecked(f.read_f32_le().await?) };
                    let node = nodes.get_mut(&node_id).unwrap();
                    node.add_neighbor(neighbor_id, layer, distance, 1024);
                    let neighbor_node = nodes.get_mut(&neighbor_id).unwrap();
                    neighbor_node.add_neighbor(node_id, layer, distance, 1024);
                }
                2 => {
                    let node_id = f.read_u64_le().await?;
                    let layer = f.read_u64_le().await?;
                    let neighbor_id = f.read_u64_le().await?;
                    let node = nodes.get_mut(&node_id).unwrap();
                    node.delete_neighbor(neighbor_id, layer);
                    let neighbor_node = nodes.get_mut(&neighbor_id).unwrap();
                    neighbor_node.delete_neighbor(node_id, layer);
                }
                3 => {
                    entry_point = Some(f.read_u64_le().await?);
                }
                4 => {
                    max_layer = Some(f.read_u64_le().await?);
                }
                _ => return Err(DBError::CorruptData("invalid token")),
            };
        }
        let nodes_vec: Vec<Node> = nodes.into_values().into_iter().collect();
        Ok(Some({
            RestoreResult {
                nodes: nodes_vec,
                entry_point,
                max_layer,
            }
        }))
    }
}
