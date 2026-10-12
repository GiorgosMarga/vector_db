use core::fmt;

#[derive(Debug)]
pub enum DBError {
    Io(std::io::Error),
    CorruptData(&'static str),
    InvalidMetadata(&'static str),
    NotFound(&'static str),
    InvalidInput(&'static str),
    Other(&'static str),
}

impl fmt::Display for DBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DBError::Io(err) => write!(f, "io error: {err}"),
            DBError::CorruptData(msg) => write!(f, "corrupt data: {msg}"),
            DBError::InvalidMetadata(msg) => write!(f, "invalid metadata: {msg}"),
            DBError::NotFound(msg) => write!(f, "not found: {msg}"),
            DBError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            DBError::Other(msg) => write!(f, "err: {msg}"),
        }
    }
}

impl std::error::Error for DBError {}

impl From<std::io::Error> for DBError {
    fn from(err: std::io::Error) -> Self {
        DBError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, DBError>;
