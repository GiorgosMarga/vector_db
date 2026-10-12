use core::num;

pub struct Replacer {
    referenced: Vec<u8>,
    is_evictable: Vec<u8>,
    cursor: usize,
    num_frames: usize,
}

impl Replacer {
    pub fn new(num_frames: usize) -> Self {
        // each byte in the vecs refer to 8 pages. bit 1 -> true & bit 0 -> false
        let vec_size = num_frames.div_ceil(8);
        let referenced = vec![0u8; vec_size];
        let is_evictable = vec![0u8; vec_size];
        Self {
            referenced,
            is_evictable,
            cursor: num_frames - 1,
            num_frames,
        }
    }

    pub fn victim(&mut self) -> Option<usize> {
        // Check at most 2 full rotations around the clock
        for _ in 0..(2 * self.num_frames) {
            let frame_id = self.cursor;

            self.cursor = if self.cursor == 0 {
                self.num_frames - 1
            } else {
                self.cursor - 1
            };

            let byte_idx = frame_id / 8;
            let mask = 1 << (frame_id % 8);

            let is_evictable = (self.is_evictable[byte_idx] & mask) != 0;
            let is_referenced = (self.referenced[byte_idx] & mask) != 0;

            if is_evictable {
                if is_referenced {
                    // Second chance: clear referenced bit
                    self.referenced[byte_idx] &= !mask;
                } else {
                    // VICTIM FOUND!
                    // Mark frame as no longer evictable:
                    self.is_evictable[byte_idx] &= !mask;
                    return Some(frame_id);
                }
            }
        }

        None // Only returns None if literally every frame has pin_count > 0!
    }

    pub fn set_evictable(&mut self, frame_id: usize, evictable: bool) {
        let byte = &mut self.is_evictable[frame_id / 8];
        let mask = 1 << (frame_id % 8);
        if evictable {
            *byte |= mask;
        } else {
            *byte &= !mask;
        }
    }

    pub fn record_access(&mut self, frame_id: usize) {
        self.referenced[frame_id / 8] |= 1 << (frame_id % 8);
    }
}
