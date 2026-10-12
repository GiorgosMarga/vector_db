use crate::errors::Result;
use crate::memgraph::graph::HNSW;
use crate::memgraph::node::Node;
use crate::storage::graph_store::GraphStore;
use crate::storage::payload_store::PayloadStore;
use crate::storage::vector_store::VectorStore;
use std::sync::{Arc, Weak};
use std::time::Duration;
use std::vec;
use tokio::sync::RwLock;
use tokio::time;

const VECTOR_DIMS: u64 = 300;

pub trait VectorProvider {
    fn get_vector(
        &self,
        node_id: u64,
    ) -> impl std::future::Future<Output = Result<Arc<[f32]>>> + Send;
}
pub trait NodeProvider {
    fn get_node(&self, node_id: u64)
    -> impl std::future::Future<Output = Result<Arc<Node>>> + Send;
}

pub struct Engine {
    memgraph: RwLock<HNSW<Arc<VectorStore>>>,
    vector_store: Arc<VectorStore>,
    payload_store: PayloadStore,
    graph_store: GraphStore,
}

impl Engine {
    pub async fn new() -> Result<Arc<Self>> {
        let vector_store = Arc::new(VectorStore::new().await?);
        let m: usize = 16;
        let engine = Arc::new(Self {
            memgraph: RwLock::new(HNSW::new(
                vector_store.clone(),
                m,
                2 * m,
                200,
                1.0 / f64::ln(m as f64),
            )),
            vector_store,
            payload_store: PayloadStore::new().await?,
            graph_store: GraphStore::new().await?,
        });
        let weak = Arc::downgrade(&engine);
        tokio::spawn(Self::checkpoint(weak));

        Ok(engine)
    }

    async fn checkpoint(weak: Weak<Self>) {
        let mut interval = time::interval(Duration::from_millis(1000));
        loop {
            interval.tick().await;
            let Some(engine) = weak.upgrade() else {
                break;
            };
            engine
                .graph_store
                .flush()
                .await
                .expect("Error flushing graph store.");
            engine
                .vector_store
                .flush()
                .expect("Error flushing vector store.");
            engine
                .payload_store
                .flush()
                .await
                .expect("Error flushing payload store.");
        }
    }
    pub async fn insert(&self, vector: &[f32], payload: &[u8]) -> Result<()> {
        let mut memgraph = self.memgraph.write().await;
        let id = memgraph.get_lateset_id();
        let _ = self.payload_store.append_payload(id, payload).await?;
        let _ = self.vector_store.append_vector(id, vector).await?;
        let records = memgraph.insert(id, vector).await?;
        self.graph_store.insert(&records).await?;

        Ok(())
    }
    pub async fn query(&self, vector: &[f32]) -> Result<Vec<String>> {
        let nodes = self.memgraph.read().await.query(vector, 5).await?;
        let mut result = vec![];
        for node in nodes {
            let payload = self.get_payload(node.1).await?.1;
            result.push(String::from_utf8(payload).unwrap());
        }
        Ok(result)
    }
    pub async fn get_payload(&self, node_id: u64) -> Result<(u64, Vec<u8>)> {
        self.payload_store.read_payload(node_id).await
    }

    pub async fn get_vector(&self, node_id: u64) -> Result<Arc<[f32]>> {
        self.vector_store.get_vector(node_id).await
    }
    pub async fn restore(&self) -> Result<()> {
        let mut memgraph = self.memgraph.write().await;
        if let Some(result) = self.graph_store.restore().await? {
            memgraph.insert_nodes(result.nodes);
            if let Some(entry_point) = result.entry_point {
                memgraph.set_entry_point(entry_point);
            }
            if let Some(max_layer) = result.max_layer {
                memgraph.set_max_layer(max_layer);
            }
        };
        Ok(())
    }
}

impl VectorProvider for Arc<VectorStore> {
    async fn get_vector(&self, node_id: u64) -> Result<Arc<[f32]>> {
        self.read_vector(node_id).await
    }
}
