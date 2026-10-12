use std::collections::HashMap;

use crate::{
    buffer_pool::{disk_manager::DiskManager, frame::Frame, replacer::Replacer},
    errors::{DBError, Result},
};

pub struct BufferPoolManagerInner {
    disk_manager: DiskManager,
    replacer: Replacer,
    frames: Vec<Frame>,
    page_table: HashMap<u64, usize>, // page_id -> frame_id
    free_list: Vec<usize>,
    pool_size: usize, // init: [0,1, ... num_frames - 1]
}

impl BufferPoolManagerInner {
    pub fn new(pool_size: usize) -> Self {
        // Pre-allocate all memory frames
        let mut frames = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            frames.push(Frame::new());
        }
        // All frames start out on the free list: [0, 1, 2, ..., pool_size - 1]
        let free_list: Vec<usize> = (0..pool_size).collect();

        Self {
            disk_manager: DiskManager::new().unwrap(),
            replacer: Replacer::new(pool_size),
            frames,
            page_table: HashMap::with_capacity(pool_size),
            free_list,
            pool_size,
        }
    }

    pub fn get_frame_mut(&mut self, frame_id: usize) -> &mut Frame {
        &mut self.frames[frame_id]
    }

    pub fn get_frame(&self, frame_id: usize) -> &Frame {
        &self.frames[frame_id]
    }
    pub fn fetch_frame_page(&mut self, page_id: u64) -> Result<usize> {
        // Cache hit
        if let Some(&frame_id) = self.page_table.get(&page_id) {
            let frame = &mut self.frames[frame_id];
            frame.pin_count += 1;
            self.replacer.set_evictable(frame_id, false); // pin it
            self.replacer.record_access(frame_id);
            return Ok(frame_id);
        }
        // Cache miss
        // check free list
        let frame_id = if let Some(free_id) = self.free_list.pop() {
            free_id
        } else if let Some(victim_id) = self.replacer.victim() {
            victim_id
        } else {
            return Err(DBError::Other(
                "Buffer pool is full and all frames are pinned",
            ));
        };

        let frame = &mut self.frames[frame_id];
        if let Some(old_page_id) = frame.page_id {
            if frame.is_dirty {
                self.disk_manager.write_page(old_page_id, &frame.data)?;
            }
            self.page_table.remove(&old_page_id);
        }
        self.disk_manager.read_page(page_id, &mut frame.data)?;

        frame.page_id = Some(page_id);
        frame.pin_count = 1;
        frame.is_dirty = false;

        // register in page_table & replacer:
        self.page_table.insert(page_id, frame_id);
        self.replacer.set_evictable(frame_id, false);
        self.replacer.record_access(frame_id);

        Ok(frame_id)
    }

    pub fn unpin_page(&mut self, page_id: u64, is_dirty: bool) -> Result<()> {
        let &frame_id = self
            .page_table
            .get(&page_id)
            .ok_or_else(|| DBError::NotFound("Page not found to unpin"))?;

        let frame = &mut self.frames[frame_id];
        if frame.pin_count > 0 {
            frame.pin_count -= 1;
        }
        if is_dirty {
            frame.is_dirty = true;
        }
        if frame.pin_count == 0 {
            self.replacer.set_evictable(frame_id, true);
        }

        Ok(())
    }
    pub fn flush_all(&mut self) -> Result<()> {
        for frame in &mut self.frames {
            if frame.is_dirty {
                if let Some(page_id) = frame.page_id {
                    self.disk_manager.write_page(page_id, &frame.data)?;
                    frame.is_dirty = false;
                }
            }
        }
        self.disk_manager.sync()
    }
}
