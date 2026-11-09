use std::sync::{Arc, Mutex, RwLockWriteGuard};

use crate::{
    buffer_pool::{bpm_inner::BufferPoolManagerInner, frame::Frame},
    page_allocator::page::Page,
};

pub const NODE_SIZE: usize = 1466; // Fixed size per node record (aligned to 64 bytes)
pub const NODES_PER_PAGE: usize = 44; // 42 * 1536 = 64,512 <= 65,536 (64 KB)
pub const VECTOR_DIMS: usize = 300;
pub const MAX_NEIGHBORS: usize = 32;
pub const PAGE_SIZE: usize = 64 * 1024;

/// A write-latched page guard that owns a raw pointer to the frame's data.
///
/// Safety: The frame is pinned (pin_count > 0) while this guard exists,
/// so the underlying memory is stable and won't be evicted.
///
/// Field drop order matters here: Rust drops fields in declaration order, so
/// `_latch` (last) is released AFTER `Drop::drop()` calls `unpin_page`.
pub struct PageGuardWrite<T> {
    bpm: Arc<Mutex<BufferPoolManagerInner>>,
    page: Page<T>,
    page_id: u64,
    is_dirty: bool,
    // Safety: The underlying RwLock lives in the Frame, which is pinned
    // (pin_count > 0) for the lifetime of this guard. The lifetime is
    // transmuted to 'static but is actually tied to the frame's lifetime.
    // This field MUST be declared last so it is dropped after Drop::drop() runs.
    _latch: RwLockWriteGuard<'static, ()>,
}

// Safety: The frame is pinned (pin_count > 0), so the memory is stable
unsafe impl<T> Send for PageGuardWrite<T> {}
unsafe impl<T> Sync for PageGuardWrite<T> {}

impl<T> PageGuardWrite<T> {
    /// Create a new write guard from a pinned frame.
    ///
    /// # Safety
    /// The caller must ensure the frame is pinned (pin_count > 0) and
    /// will remain valid for the lifetime of this guard.
    pub fn new(bpm: Arc<Mutex<BufferPoolManagerInner>>, frame: &Frame) -> Self {
        let latch = frame.latch.write().unwrap();
        // Safety: frame is pinned (pin_count > 0) so the RwLock will not be
        // deallocated while this guard exists.
        let latch: RwLockWriteGuard<'static, ()> = unsafe { std::mem::transmute(latch) };
        let data = frame.data.as_ptr() as *mut u8;
        Self {
            page: Page::<T>::new(data),
            page_id: frame.page_id.unwrap(),
            is_dirty: false,
            bpm,
            _latch: latch,
        }
    }

    pub fn get_data(&self) -> &[u8; PAGE_SIZE] {
        self.page.get_data()
    }

    pub fn get_data_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        self.is_dirty = true;
        self.page.get_data_mut()
    }

    pub fn mark_dirty(&mut self) {
        self.is_dirty = true;
    }

    pub fn get_page(&self) -> &Page<T> {
        &self.page
    }

    pub fn get_page_mut(&mut self) -> &mut Page<T> {
        self.is_dirty = true;
        &mut self.page
    }
}

impl<T> Drop for PageGuardWrite<T> {
    fn drop(&mut self) {
        // Automatically decrements pin_count when guard goes out of scope!
        if let Ok(mut bpm) = self.bpm.lock() {
            let _ = bpm.unpin_page(self.page_id, self.is_dirty);
        }
    }
}
