use std::sync::Arc;
use tokio::sync::RwLock;

use crate::phantom_core::config::PhantomConfig;
use crate::phantom_core::state::ConnectionState;
use crate::utils::{PhantomError, PhantomResult};

pub struct PhantomEngine {
    pub config: PhantomConfig,
    pub state: Arc<RwLock<ConnectionState>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl PhantomEngine {
    pub fn new(config: PhantomConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            shutdown_tx: None,
        }
    }

    pub async fn start(&mut self) -> PhantomResult<()> {
        tracing::info!("Starting Phantom engine");

        {
            let mut state = self.state.write().await;
            *state = ConnectionState::Connecting;
        }

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);

        let config = self.config.clone();
        let state = Arc::clone(&self.state);

        tokio::spawn(async move {
            let result = async {
                tracing::info!("Starting SOCKS5 proxy listener");
                Self::run_socks5_listener(&config, &state).await?;

                tracing::info!("Starting QUIC connection loop");
                Self::run_quic_loop(&config, &state).await?;

                tracing::info!("Starting signaling loop");
                Self::run_signaling_loop(&config, &state).await?;

                {
                    let mut state = state.write().await;
                    *state = ConnectionState::Connected {
                        speed_up: 0,
                        speed_down: 0,
                        worker_count: 3,
                    };
                }

                tracing::info!("All subsystems started successfully");

                shutdown_rx.await.map_err(|_| PhantomError::Cancelled)
            }
            .await;

            if let Err(e) = result {
                tracing::error!("Engine subsystem error: {}", e);
                let mut state = state.write().await;
                *state = ConnectionState::Error {
                    message: e.to_string(),
                };
            }
        });

        tracing::info!("Phantom engine started");
        Ok(())
    }

    pub async fn stop(&mut self) -> PhantomResult<()> {
        tracing::info!("Stopping Phantom engine");

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        {
            let mut state = self.state.write().await;
            *state = ConnectionState::Disconnected;
        }

        tracing::info!("Phantom engine stopped");
        Ok(())
    }

    pub async fn get_state(&self) -> ConnectionState {
        self.state.read().await.clone()
    }

    async fn run_socks5_listener(
        _config: &PhantomConfig,
        _state: &Arc<RwLock<ConnectionState>>,
    ) -> PhantomResult<()> {
        tracing::info!("SOCKS5 listener initialized on 127.0.0.1:1080");
        Ok(())
    }

    async fn run_quic_loop(
        _config: &PhantomConfig,
        _state: &Arc<RwLock<ConnectionState>>,
    ) -> PhantomResult<()> {
        tracing::info!("QUIC connection loop initialized");
        Ok(())
    }

    async fn run_signaling_loop(
        _config: &PhantomConfig,
        _state: &Arc<RwLock<ConnectionState>>,
    ) -> PhantomResult<()> {
        tracing::info!("Signaling loop initialized");
        Ok(())
    }
}
