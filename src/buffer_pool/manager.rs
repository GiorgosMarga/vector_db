use std::sync::{Arc, Mutex};

use crate::{
    buffer_pool::{
        bpm_inner::{self, BufferPoolManagerInner},
        page_guard::PageGuard,
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
    pub fn fetch_page(&self, page_id: u64) -> Result<PageGuard> {
        let mut inner = self.inner.lock().unwrap();
        let frame_id = inner.fetch_frame_page(page_id)?;
        let data_ptr = inner.get_frame(frame_id).data.as_ptr();

        Ok(PageGuard::new(
            self.inner.clone(),
            page_id,
            frame_id,
            data_ptr,
        ))
    }

    pub fn flush_all(&self) -> Result<()> {
        self.inner.lock().unwrap().flush_all()
    }
}
