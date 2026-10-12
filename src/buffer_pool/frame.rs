pub struct Frame {
    pub data: Box<[u8]>,      // Exactly PAGE_SIZE
    pub page_id: Option<u64>, // Which disk page is currently here?
    pub pin_count: usize,     // How many active readers?
    pub is_dirty: bool,       // Was it writter to?
}

impl Frame {
    pub fn new() -> Self {
        Self {
            data: vec![0u8; 64 * 1024].into_boxed_slice(),
            page_id: None,
            pin_count: 0,
            is_dirty: false,
        }
    }
}
