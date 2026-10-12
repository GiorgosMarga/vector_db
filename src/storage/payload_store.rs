use std::{
    io::SeekFrom::{End, Start},
    sync::atomic::{AtomicBool, Ordering::Relaxed},
};

use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufWriter},
    sync::Mutex,
};

use crate::errors::Result;

pub struct PayloadStore {
    f: Mutex<BufWriter<File>>,
    f_index: Mutex<BufWriter<File>>,
    path: &'static str,
    offset_path: &'static str,
    dirty: AtomicBool,
}

impl PayloadStore {
    pub async fn new() -> Result<Self> {
        let path = "db/payload.bin";
        let offset_path = "db/payload_index.bin";
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        let f_index = OpenOptions::new()
            .create(true)
            .append(true)
            .open(offset_path)
            .await?;
        Ok(Self {
            f: Mutex::new(BufWriter::new(f)),
            f_index: Mutex::new(BufWriter::new(f_index)),
            path,
            offset_path,
            dirty: AtomicBool::new(false),
        })
    }

    pub async fn append_payload(&self, id: u64, payload: &[u8]) -> Result<u64> {
        let mut f = self.f.lock().await;
        let offset = f.seek(End(0)).await?;

        let mut buf = Vec::with_capacity(2 * size_of::<u64>() + payload.len());
        buf.write_u64_le(id).await?;
        buf.write_u64_le(payload.len() as u64).await?;
        buf.extend_from_slice(payload);

        f.write_all(&buf).await?;
        self.f_index.lock().await.write_u64_le(offset).await?;
        self.dirty.store(true, Relaxed);

        Ok(offset)
    }

    pub async fn flush(&self) -> Result<()> {
        if self.dirty.swap(false, Relaxed) {
            {
                let mut f = self.f.lock().await;
                f.flush().await?;
                f.get_ref().sync_all().await?;
            }
            let mut f = self.f_index.lock().await;
            f.flush().await?;
            f.get_ref().sync_all().await?;
        }
        Ok(())
    }

    pub async fn read_payload(&self, node_id: u64) -> Result<(u64, Vec<u8>)> {
        let index_offset = node_id * size_of::<u64>() as u64;
        let mut f = OpenOptions::new().read(true).open(self.offset_path).await?;
        f.seek(Start(index_offset)).await?;
        let offset = f.read_u64_le().await?;

        let mut f = OpenOptions::new().read(true).open(self.path).await?;
        f.seek(Start(offset)).await?;
        let mut buf = [0; 2 * size_of::<u64>()];
        f.read_exact(&mut buf).await?;

        let id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let size = u64::from_le_bytes(buf[8..16].try_into().unwrap());

        let mut buf = vec![0; size as usize];
        f.read_exact(&mut buf).await?;
        Ok((id, buf))
    }
}
