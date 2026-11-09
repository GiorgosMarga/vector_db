use std::sync::RwLock;

pub struct Frame {
    pub data: Box<[u8; 64 * 1024]>, // Exactly PAGE_SIZE
    pub page_id: Option<u64>,       // Which disk page is currently here?
    pub pin_count: usize,           // How many active readers?
    pub is_dirty: bool,             // Was it writter to?
    pub latch: RwLock<()>,
}

impl Frame {
    pub fn new() -> Self {
        Self {
            latch: RwLock::new(()),
            data: Box::new([0u8; 64 * 1024]),
            page_id: None,
            pin_count: 0,
            is_dirty: false,
        }
    }
}
