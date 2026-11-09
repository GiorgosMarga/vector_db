use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub struct PageAllocator {
    next_page_id: AtomicU64,
}

impl PageAllocator {
    pub fn new(last_page_id: u64) -> Self {
        Self {
            next_page_id: AtomicU64::new(last_page_id + 1),
        }
    }

    pub fn page_alloc(&self) -> u64 {
        self.next_page_id.fetch_add(1, Relaxed)
    }
}
