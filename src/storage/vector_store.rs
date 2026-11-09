use std::sync::Arc;

use crate::{
    buffer_pool::{
        manager::BufferPoolManager,
        page_guard::{NODE_SIZE, NODES_PER_PAGE},
    },
    errors::Result,
    memgraph::node::Node,
};

pub struct VectorStore {
    bpm: BufferPoolManager,
}

impl VectorStore {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            bpm: BufferPoolManager::new(500),
        })
    }
}
