use crate::DIMS;
use crate::buffer_pool::manager::BufferPoolManager;
use crate::buffer_pool::page_guard::PageGuardWrite;
use crate::errors::Result;
use crate::memgraph::graph::HNSW;
use crate::page_allocator::page::{
    self, DBHeaderPage, SegmentDirectoryPage, SegmentMetadata, TranslationPage, VectorPage,
};
use crate::storage::page_allocator::PageAllocator;
use crate::storage::payload_store::PayloadStore;
use crate::storage::vector_store::VectorStore;
use crate::storage::wal::{Wal, WalRecord};
use core::num;
use std::sync::{Arc, Weak};
use std::time::Duration;
use std::vec;
use tokio::sync::RwLock;
use tokio::time;

const VECTOR_DIMS: u64 = 300;
const VEC_SIZE_LIMIT: usize = 50_000;
const DB_PAGE_ID: u64 = 1;
const DIRECTORY_ID: u64 = 2;
const MAX_METADATA_CTR: u64 = 52;

pub struct Engine {
    memgraph: RwLock<HNSW>,
    router_graph: RwLock<HNSW>,
    vector_store: VectorStore,
    payload_store: PayloadStore,
    page_allocator: PageAllocator,
    buffer_pool: BufferPoolManager,
    header_guard: RwLock<PageGuardWrite<DBHeaderPage>>,
    wal: Wal,
}

impl Engine {
    pub async fn new() -> Result<Arc<Self>> {
        let bpm = BufferPoolManager::new(4096);
        let mut guard = bpm.fetch_page::<DBHeaderPage>(0)?;
        let page = guard.get_page_mut();
        let mut header = page.get_header();

        match header.magic {
            magic if magic == [0u8; 8] => {
                page.set_header(*b"VECTORDB", 1, 64 * 1024, 1, 2, 0);
                let mut segment_page_guard = bpm.fetch_page::<SegmentDirectoryPage>(1)?;
                segment_page_guard.get_page_mut().set_page_id(1);
                header = page.get_header();
            }
            magic if magic == *b"VECTORDB" => {
                println!("Already initialized");
            }
            _ => {
                println!("Invalid magic number");
            }
        }

        println!("{:?}", header);
        let page_allocator = PageAllocator::new(header.last_page_id);

        let m: usize = 16;
        let engine = Arc::new(Self {
            buffer_pool: bpm,
            page_allocator,
            header_guard: RwLock::new(guard),
            memgraph: RwLock::new(HNSW::new(
                m,
                2 * m,
                200,
                1.0 / f64::ln(m as f64),
                header.next_vector_id,
            )),
            router_graph: RwLock::new(HNSW::new(
                m,
                2 * m,
                200,
                1.0 / f64::ln(m as f64),
                header.next_vector_id,
            )),
            vector_store: VectorStore::new().await?,
            payload_store: PayloadStore::new().await?,
            wal: Wal::new().await?,
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
                .buffer_pool
                .flush_all()
                .expect("[ERROR]: flushing buffer pool");
            engine.wal.flush().await.expect("[ERROR]: wal flushing");
        }
    }
    pub async fn insert(&self, vector: &[f32], payload: &[u8]) -> Result<()> {
        let mut memgraph = self.memgraph.write().await;
        let id = memgraph.get_lateset_id();
        self.wal.log_insert(id, vector, payload).await?;
        memgraph.insert(id, vector).await?;
        self.payload_store.append_payload(id, payload).await?;
        if memgraph.size() == VEC_SIZE_LIMIT {
            let memgraph_to_seal = std::mem::replace(
                &mut *memgraph,
                HNSW::new(16, 32, 200, 1.0 / f64::ln(32 as f64), 50_000),
            );
            drop(memgraph);

            self.seal_batch(memgraph_to_seal).await?;
        }
        Ok(())
    }

    async fn seal_batch(&self, memgraph: HNSW) -> Result<()> {
        let nodes = memgraph.collect_bfs_nodes();
        let mut id_map = vec![0; VEC_SIZE_LIMIT];

        let mut first_vector_page: Option<u64> = None;
        let mut data_page_count: u32 = 0;
        let mut prev_vector_page: Option<u64> = None;
        let mut centroid = [0f32; DIMS];
        for (idx, chunk) in nodes.chunks(44).enumerate() {
            let new_page_id = self.allocate_page().await;
            if idx == 0 {
                first_vector_page = Some(new_page_id);
            }
            {
                let mut guard = self.buffer_pool.fetch_page::<VectorPage>(new_page_id)?;
                let page = guard.get_page_mut();
                for (slot, &node) in chunk.iter().enumerate() {
                    for (i, val) in node.vec.iter().enumerate() {
                        centroid[i] += val;
                    }
                    page.write_node(slot, node)?;
                }
                data_page_count += 1;
            }
            if let Some(prev) = prev_vector_page {
                let mut prev_guard = self.buffer_pool.fetch_page::<VectorPage>(new_page_id)?;
                prev_guard.get_page_mut().set_next_page_id(prev);
            }
            prev_vector_page = Some(new_page_id);
        }
        for (new_id, node) in nodes.iter().enumerate() {
            id_map[node.id as usize] = new_id as u64;
        }

        let mut prev_page_id: Option<u64> = None;
        let mut first_page_id: Option<u64> = None;

        for (chunk_id, ids) in id_map.chunks(2049).enumerate() {
            let new_page_id = self.allocate_page().await;
            if chunk_id == 0 {
                first_page_id = Some(new_page_id);
            }

            let mut guard = self
                .buffer_pool
                .fetch_page::<TranslationPage>(new_page_id)?;
            let page = guard.get_page_mut();
            page.set_nodes_map(ids);
            page.set_ids_count(ids.len() as u16);

            if let Some(prev_id) = prev_page_id {
                let mut prev_guard = self.buffer_pool.fetch_page::<TranslationPage>(prev_id)?;
                prev_guard
                    .get_page_mut()
                    .set_next_translation_page(new_page_id);
            }
            prev_page_id = Some(new_page_id);
        }

        for i in 0..DIMS {
            centroid[i] /= VEC_SIZE_LIMIT as f32;
        }

        let metadata = SegmentMetadata {
            centroid,
            data_page_count,
            entry_node_id: memgraph.get_entry_point().unwrap(),
            first_data_page: first_vector_page.unwrap(),
            first_translation_page: first_page_id.unwrap(),
            segment_id: 1,
            translation_page_count: 1163,
            vector_count: 50_000,
        };

        let mut guard = self.fetch_directory_page().await?;
        let page = guard.get_page_mut();
        println!("appending: {:?} to {}", metadata, page.get_page_id());
        page.append_entry(metadata);

        self.router_graph.write().await.insert(1, &centroid).await?;
        Ok(())
    }

    async fn fetch_directory_page(&self) -> Result<PageGuardWrite<SegmentDirectoryPage>> {
        let mut page_id = {
            self.header_guard
                .read()
                .await
                .get_page()
                .get_segment_directory_page_id()
        };
        loop {
            let mut guard = self
                .buffer_pool
                .fetch_page::<SegmentDirectoryPage>(page_id)?;
            let page = guard.get_page_mut();
            if page.get_entries_ctr() < MAX_METADATA_CTR as u32 {
                return Ok(guard);
            }
            page_id = page.get_next_page_id();
            if page_id == 0 {
                let new_page_id = self.allocate_page().await;
                let mut new_guard = self
                    .buffer_pool
                    .fetch_page::<SegmentDirectoryPage>(new_page_id)?;
                page.set_next_page_id(new_page_id);
                new_guard.get_page_mut().set_page_id(new_page_id);
                self.header_guard
                    .write()
                    .await
                    .get_page_mut()
                    .set_segment_directory_page_id(new_page_id);
                return Ok(new_guard);
            }
        }
    }
    pub async fn query(&self, vector: &[f32]) -> Result<Vec<String>> {
        let nodes = self.memgraph.read().await.query(vector, 5).await?;

        let router_nodes = self.router_graph.read().await.query(vector, 5).await?;

        for router_node in router_nodes {
            let segment_id = router_node.1;
            let segment_page_id = self.find_segment_page_id(segment_id).await?;
            if segment_page_id == 0 {
                println!("[ERROR]: cant locate segment");
                break;
            }
            let metadata = self
                .get_segment_metadata(segment_page_id, segment_id)
                .await?;
            let guard = self
                .buffer_pool
                .fetch_page::<VectorPage>(metadata.first_data_page)?;
            let vector_page = guard.get_page();
          }

        let segments_page_id = {
            self.header_guard
                .read()
                .await
                .get_page()
                .get_segment_directory_page_id()
        };
        let mut result = vec![];
        for node in nodes {
            let payload = self.get_payload(node.1).await?.1;
            result.push(String::from_utf8(payload).unwrap());
        }

        let guard = self
            .buffer_pool
            .fetch_page::<SegmentDirectoryPage>(segments_page_id)?;

        let page = guard.get_page();
        println!("Page: {}", page.get_entries_ctr());
        Ok(result)
    }
    pub async fn get_payload(&self, node_id: u64) -> Result<(u64, Vec<u8>)> {
        self.payload_store.read_payload(node_id).await
    }

    pub async fn get_vector(&self, node_id: u64) -> Result<Arc<[f32]>> {
        // self.vector_store.read_vector(node_id).await
        _ = node_id;
        Ok(Arc::from([0f32; 300]))
    }
    pub async fn restore(&self) -> Result<()> {
        let mut memgraph = self.memgraph.write().await;
        let records = self.wal.restore().await?;
        for record in records {
            match record {
                WalRecord::Insert {
                    id,
                    vector,
                    payload,
                } => {
                    self.payload_store.append_payload(id, &payload).await?;
                    memgraph.insert(id, &vector).await?;
                }
                WalRecord::Delete { id: _ } => {}
            }
        }
        Ok(())
    }

    async fn find_segment_page_id(&self, segment_id: u64) -> Result<u64> {
        let mut segment_page_id = {
            self.header_guard
                .read()
                .await
                .get_page()
                .get_segment_directory_page_id()
        };
        let mut iter = 0;

        println!("Searching for segment_id: {}", segment_id);
        loop {
            let guard = self
                .buffer_pool
                .fetch_page::<SegmentDirectoryPage>(segment_page_id)?;
            let page = guard.get_page();

            let num_segments = page.get_entries_ctr() as u64;

            println!(
                "Segment page: {} has segments [{},{})",
                segment_page_id,
                iter * 52,
                (iter + 1) * num_segments
            );

            if segment_id >= iter * 52 && segment_id <= (iter + 1) * num_segments {
                return Ok(page.get_page_id());
            }

            segment_page_id = page.get_next_page_id();
            if segment_page_id == 0 {
                return Ok(0);
            }
            iter += 1;
        }
    }

    async fn get_segment_metadata(
        &self,
        segment_page_id: u64,
        segment_id: u64,
    ) -> Result<SegmentMetadata> {
        let guard = self
            .buffer_pool
            .fetch_page::<SegmentDirectoryPage>(segment_page_id)?;
        let page = guard.get_page();
        let metadata = page.get_entry(segment_id).unwrap();
        Ok(metadata)
    }

    async fn allocate_page(&self) -> u64 {
        let new_page = self.page_allocator.page_alloc();
        let mut guard = self.header_guard.write().await;
        let page = guard.get_page_mut();
        page.set_last_page_id(new_page);
        new_page
    }
}
