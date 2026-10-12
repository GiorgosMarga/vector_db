use std::sync::{Arc, Mutex};

use crate::buffer_pool::bpm_inner::BufferPoolManagerInner;

pub struct PageGuard {
    bpm: Arc<Mutex<BufferPoolManagerInner>>,
    pub page_id: u64,
    pub frame_id: usize,
    data_ptr: *const u8,
    is_dirty: bool,
}
// Safety: The frame is pinned (pin_count > 0), so the memory is stable
unsafe impl Send for PageGuard {}
unsafe impl Sync for PageGuard {}

impl PageGuard {
    pub fn new(
        bpm: Arc<Mutex<BufferPoolManagerInner>>,
        page_id: u64,
        frame_id: usize,
        data_ptr: *const u8,
    ) -> Self {
        Self {
            bpm,
            page_id,
            frame_id,
            data_ptr,
            is_dirty: false,
        }
    }
    pub fn read_data(&self, slot: usize) -> &[f32] {
        let offset = slot * 1200;
        // SAFETY:
        // offset is a multiple of 4, so the pointer is 4-byte aligned for f32.
        // 1200 bytes = exactly 300 floats.
        // The slice lifetime is bound to `&self` (the PageGuard).
        unsafe {
            let float_ptr = self.data_ptr.add(offset) as *const f32;
            std::slice::from_raw_parts(float_ptr, 300)
        }
    }

    /// Write a 300-dimension vector to slot (0..54)                                                                                                        
    pub fn write_data(&mut self, slot: usize, vector: &[f32]) {
        self.is_dirty = true;
        let offset = slot * 1200;

        unsafe {
            let dest = self.data_ptr as *mut u8;
            let dest_slice = std::slice::from_raw_parts_mut(dest.add(offset), 1200);
            for (i, &val) in vector.iter().enumerate() {
                dest_slice[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }
    }
}
impl Drop for PageGuard {
    fn drop(&mut self) {
        // Automatically decrements pin_count when guard goes out of scope!
        if let Ok(mut bpm) = self.bpm.lock() {
            let _ = bpm.unpin_page(self.page_id, self.is_dirty);
        }
    }
}
