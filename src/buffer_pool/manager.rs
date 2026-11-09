use std::sync::{Arc, Mutex};

use crate::{
    buffer_pool::{
        bpm_inner::{self, BufferPoolManagerInner},
        page_guard::PageGuardWrite,
    },
    errors::Result,
};

pub struct BufferPoolManager {
    inner: Arc<Mutex<bpm_inner::BufferPoolManagerInner>>,
}

impl BufferPoolManager {
    pub fn new(pool_size: usize) -> Self {
        let bpm = BufferPoolManagerInner::new(pool_size);
        Self {
            inner: Arc::new(Mutex::new(bpm)),
        }
    }
    pub fn fetch_page<T>(&self, page_id: u64) -> Result<PageGuardWrite<T>> {
        let mut inner = self.inner.lock().unwrap();
        let frame_id = inner.fetch_frame_page(page_id)?;
        let frame = inner.get_frame(frame_id);

        Ok(PageGuardWrite::new(self.inner.clone(), frame))
    }

    pub fn flush_all(&self) -> Result<()> {
        self.inner.lock().unwrap().flush_all()
    }
}
