use std::{
    collections::HashMap,
    io::SeekFrom::Start,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
};

use tokio::{
    fs::{self, File},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufWriter},
    sync::{Mutex, RwLock},
};

use crate::{
    DIMS,
    buffer_pool::{self, manager::BufferPoolManager},
    errors::{DBError, Result},
};

const VECTORS_PER_PAGE: u64 = 54;
const VECTOR_SIZE: usize = 300 * std::mem::size_of::<f32>();

pub struct VectorStore {
    bpm: BufferPoolManager,
}

impl VectorStore {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            bpm: BufferPoolManager::new(500),
        })
    }

    pub async fn append_vector(&self, node_id: u64, vector: &[f32]) -> Result<()> {
        let (page_id, byte_offset) = Self::locate_vector(node_id);
        let mut guard = self.bpm.fetch_page(page_id)?;
        guard.write_data(byte_offset, vector);
        Ok(())
    }
    pub async fn read_vector(&self, node_id: u64) -> Result<Arc<[f32]>> {
        let (page_id, slot) = Self::locate_vector(node_id);

        let guard = self.bpm.fetch_page(page_id)?;

        // 2. Parse into f32 floats:
        let data = guard.read_data(slot);

        Ok(Arc::from(data))
    }

    pub fn flush(&self) -> Result<()> {
        self.bpm.flush_all()
    }

    pub fn locate_vector(node_id: u64) -> (u64, usize) {
        let page_id = node_id / VECTORS_PER_PAGE;
        let slot = (node_id % VECTORS_PER_PAGE) as usize;
        (page_id, slot)
    }
}
