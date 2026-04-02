use thiserror::Error;

#[derive(Error, Debug)]
pub enum PhantomError {
    #[error("QUIC error: {0}")]
    QuicError(#[from] quinn::ConnectionError),

    #[error("KCP error: {0}")]
    KcpError(String),

    #[error("Crypto error: {0}")]
    CryptoError(String),

    #[error("SOCKS error: {0}")]
    SocksError(#[from] tokio_socks::Error),

    #[error("Signaling error: {0}")]
    SignalingError(String),

    #[error("Relay error: {0}")]
    RelayError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Connection lost: {0}")]
    ConnectionLost(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("HTTP/3 error: {0}")]
    Http3Error(String),

    #[error("Task cancelled")]
    Cancelled,
}

pub type PhantomResult<T> = Result<T, PhantomError>;
