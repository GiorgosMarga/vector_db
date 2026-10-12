use crate::errors::Result;
use std::{
    fs::{File, OpenOptions},
    os::unix::fs::FileExt,
    sync::Arc,
};

pub struct DiskManager {
    file: Arc<File>,
    path: &'static str,
}

const PAGE_SIZE: u64 = 64 * 1024; // 64kb

impl DiskManager {
    pub fn new() -> Result<Self> {
        let path = "./db/db.bin";
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        Ok(Self {
            file: Arc::new(file),
            path,
        })
    }

    pub fn read_page(&self, page_id: u64, page_data: &mut [u8]) -> Result<()> {
        assert_eq!(page_data.len(), PAGE_SIZE as usize);
        let offset = page_id * PAGE_SIZE;
        let file_len = self.file.metadata()?.len();
        // if this page hasn't been written to disk yet:
        if offset >= file_len {
            page_data.fill(0);
            return Ok(());
        }
        self.file.read_exact_at(page_data, offset)?;
        Ok(())
    }

    pub fn write_page(&self, page_id: u64, page_data: &[u8]) -> Result<()> {
        assert_eq!(page_data.len(), PAGE_SIZE as usize);
        let offset = page_id * PAGE_SIZE;
        self.file.write_all_at(page_data, offset)?;
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}
