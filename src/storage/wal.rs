use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter},
    sync::Mutex,
};

use crate::errors::{DBError, Result};

pub enum WalRecord {
    Insert {
        id: u64,
        vector: Vec<f32>,
        payload: Vec<u8>,
    },
    Delete {
        id: u64,
    },
}
pub struct Wal {
    f: Mutex<BufWriter<File>>,
    path: &'static str,
}

impl Wal {
    pub async fn new() -> Result<Self> {
        let path = "db/wal.bin";
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            f: Mutex::new(BufWriter::new(file)),
            path,
        })
    }

    pub async fn log_insert(&self, node_id: u64, vector: &[f32], payload: &[u8]) -> Result<()> {
        let mut wal = self.f.lock().await;

        wal.write_u8(0).await?;
        wal.write_u64_le(node_id).await?;
        wal.write_u16_le(vector.len() as u16).await?;
        for &v in vector.iter() {
            wal.write_f32_le(v).await?;
        }
        wal.write_u16_le(payload.len() as u16).await?;
        wal.write_all(payload).await?;
        Ok(())
    }
    pub async fn log_delete(&self, node_id: u64) -> Result<()> {
        let mut wal = self.f.lock().await;
        wal.write_u8(1).await?;
        wal.write_u64_le(node_id).await?;
        Ok(())
    }

    pub async fn flush(&self) -> Result<()> {
        let mut wal = self.f.lock().await;
        wal.flush().await?;
        wal.get_ref().sync_all().await?;
        Ok(())
    }

    pub async fn restore(&self) -> Result<Vec<WalRecord>> {
        {
            let mut wal = self.f.lock().await;
            wal.flush().await?;
        }

        let file = match OpenOptions::new().read(true).open(self.path).await {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let metadata = file.metadata().await?;
        if metadata.len() == 0 {
            return Ok(Vec::new());
        }

        let mut reader = BufReader::new(file);
        let mut records = Vec::new();

        while let Ok(tag) = reader.read_u8().await {
            match tag {
                0 => {
                    let id = reader.read_u64_le().await?;
                    let vec_len = reader.read_u16_le().await? as usize;
                    let mut vector = Vec::with_capacity(vec_len);
                    for _ in 0..vec_len {
                        vector.push(reader.read_f32_le().await?);
                    }
                    let payload_len = reader.read_u16_le().await? as usize;
                    let mut payload = vec![0u8; payload_len];
                    reader.read_exact(&mut payload).await?;

                    records.push(WalRecord::Insert {
                        id,
                        vector,
                        payload,
                    });
                }
                1 => {
                    let id = reader.read_u64_le().await?;
                    records.push(WalRecord::Delete { id });
                }
                _ => return Err(DBError::CorruptData("invalid wal tag token")),
            }
        }

        Ok(records)
    }
}
