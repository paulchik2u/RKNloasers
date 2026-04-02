use quinn::Connection;
use std::sync::{Arc, RwLock};

use crate::phantom_core::state::ConnectionState;
use crate::utils::{PhantomError, PhantomResult};

pub struct QuicConnection {
    connection: Connection,
    state: Arc<RwLock<ConnectionState>>,
}

impl QuicConnection {
    pub fn new(connection: Connection, state: Arc<RwLock<ConnectionState>>) -> Self {
        let remote = connection.remote_address();
        tracing::info!(%remote, "QUIC connection established");
        Self { connection, state }
    }

    pub async fn send(&self, data: &[u8]) -> PhantomResult<()> {
        let mut send = self
            .connection
            .open_uni()
            .await
            .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to open send stream: {}", e)))?;

        send.write_all(data)
            .await
            .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to write data: {}", e)))?;

        send.finish()
            .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to finish send stream: {}", e)))?;

        tracing::debug!(bytes = data.len(), "Data sent over QUIC stream");
        Ok(())
    }

    pub async fn recv(&self, buf: &mut [u8]) -> PhantomResult<usize> {
        let mut recv = self
            .connection
            .accept_uni()
            .await
            .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to accept receive stream: {}", e)))?;

        let n = recv
            .read(buf)
            .await
            .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to read data: {}", e)))?
            .ok_or_else(|| PhantomError::ConnectionLost("Stream closed before data received".to_string()))?;

        tracing::debug!(bytes = n, "Data received over QUIC stream");
        Ok(n)
    }

    pub async fn close(&self) {
        let remote = self.connection.remote_address();
        self.connection.close(0u32.into(), b"done");
        tracing::info!(%remote, "QUIC connection closed");

        if let Ok(mut state) = self.state.write() {
            *state = ConnectionState::Disconnected;
        }
    }

    pub fn is_alive(&self) -> bool {
        !self.connection.close_reason().is_some()
    }

    pub fn connection(&self) -> &Connection {
        &self.connection
    }
}
