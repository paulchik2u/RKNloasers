use std::sync::{Arc, RwLock};

use crate::phantom_core::config::TransportConfig;
use crate::phantom_core::state::ConnectionState;
use crate::utils::{PhantomError, PhantomResult};

use super::connection::QuicConnection;
use super::endpoint::QuicEndpoint;

pub struct QuicClient {
    endpoint: QuicEndpoint,
    connection: Arc<RwLock<Option<QuicConnection>>>,
    config: TransportConfig,
    state: Arc<RwLock<ConnectionState>>,
}

impl QuicClient {
    pub fn new(config: TransportConfig, state: Arc<RwLock<ConnectionState>>) -> PhantomResult<Self> {
        tracing::info!(protocol = %config.protocol, "Initializing QUIC client");

        let endpoint = QuicEndpoint::new()?;

        Ok(Self {
            endpoint,
            connection: Arc::new(RwLock::new(None)),
            config,
            state,
        })
    }

    pub async fn connect(&self, worker_url: &str) -> PhantomResult<()> {
        tracing::info!(url = worker_url, "Connecting to worker");

        {
            let mut state = self.state.write().map_err(|e| {
                PhantomError::ConnectionFailed(format!("Failed to acquire state lock: {}", e))
            })?;
            *state = ConnectionState::Connecting;
        }

        let server_name = worker_url
            .split("://")
            .nth(1)
            .unwrap_or(worker_url)
            .split(':')
            .next()
            .unwrap_or(worker_url)
            .to_string();

        let server_addr = worker_url
            .parse::<std::net::SocketAddr>()
            .map_err(|e| PhantomError::ConnectionFailed(format!("Invalid worker URL '{}': {}", worker_url, e)))?;

        let connection = self
            .endpoint
            .endpoint()
            .connect(server_addr, &server_name)
            .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to initiate connection: {}", e)))?
            .await
            .map_err(|e| PhantomError::ConnectionFailed(format!("Connection handshake failed: {}", e)))?;

        let quic_connection = QuicConnection::new(connection, Arc::clone(&self.state));

        {
            let mut conn_guard = self.connection.write().map_err(|e| {
                PhantomError::ConnectionFailed(format!("Failed to acquire connection lock: {}", e))
            })?;
            *conn_guard = Some(quic_connection);
        }

        {
            let mut state = self.state.write().map_err(|e| {
                PhantomError::ConnectionFailed(format!("Failed to acquire state lock: {}", e))
            })?;
            *state = ConnectionState::Connected {
                speed_up: 0,
                speed_down: 0,
                worker_count: 1,
            };
        }

        tracing::info!(url = worker_url, "Successfully connected to worker");
        Ok(())
    }

    pub async fn disconnect(&self) -> PhantomResult<()> {
        tracing::info!("Disconnecting QUIC client");

        let conn = {
            let mut guard = self.connection.write().map_err(|e| {
                PhantomError::ConnectionFailed(format!("Failed to acquire connection lock: {}", e))
            })?;
            guard.take()
        };

        if let Some(conn) = conn {
            conn.close().await;
        }

        {
            let mut state = self.state.write().map_err(|e| {
                PhantomError::ConnectionFailed(format!("Failed to acquire state lock: {}", e))
            })?;
            *state = ConnectionState::Disconnected;
        }

        tracing::info!("QUIC client disconnected");
        Ok(())
    }

    pub async fn send(&self, data: &[u8]) -> PhantomResult<()> {
        let conn_guard = self.connection.read().map_err(|e| {
            PhantomError::ConnectionFailed(format!("Failed to acquire connection lock: {}", e))
        })?;

        let conn = conn_guard
            .as_ref()
            .ok_or_else(|| PhantomError::ConnectionFailed("No active connection".to_string()))?;

        conn.send(data).await
    }

    pub async fn recv(&self) -> PhantomResult<Vec<u8>> {
        let conn_guard = self.connection.read().map_err(|e| {
            PhantomError::ConnectionFailed(format!("Failed to acquire connection lock: {}", e))
        })?;

        let conn = conn_guard
            .as_ref()
            .ok_or_else(|| PhantomError::ConnectionFailed("No active connection".to_string()))?;

        let mut buf = vec![0u8; self.config.mtu as usize];
        let n = conn.recv(&mut buf).await?;
        buf.truncate(n);
        Ok(buf)
    }

    pub async fn reconnect(&self) -> PhantomResult<()> {
        tracing::info!("Attempting to reconnect");

        self.disconnect().await?;

        let worker_url = format!("{}:{}", "127.0.0.1", 443);
        self.connect(&worker_url).await
    }

    pub async fn run_connection_loop(&self) {
        let reconnect_delay = std::time::Duration::from_secs(5);

        loop {
            let alive = {
                let conn_guard = match self.connection.read() {
                    Ok(guard) => guard,
                    Err(e) => {
                        tracing::error!("Failed to acquire connection lock: {}", e);
                        tokio::time::sleep(reconnect_delay).await;
                        continue;
                    }
                };

                conn_guard
                    .as_ref()
                    .map(|c| c.is_alive())
                    .unwrap_or(false)
            };

            if !alive {
                tracing::warn!("Connection lost, attempting reconnect in {:?}", reconnect_delay);
                tokio::time::sleep(reconnect_delay).await;

                if let Err(e) = self.reconnect().await {
                    tracing::error!(error = %e, "Reconnection failed");

                    {
                        if let Ok(mut state) = self.state.write() {
                            *state = ConnectionState::Error {
                                message: format!("Reconnection failed: {}", e),
                            };
                        }
                    }
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }
}
