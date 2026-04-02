use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::utils::error::{PhantomError, PhantomResult};
use crate::signaling::doh::DohDiscovery;
use crate::signaling::nostr::NostrDiscovery;

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Failed,
}

pub struct WorkerDiscovery {
    nostr: NostrDiscovery,
    doh: DohDiscovery,
    workers: Arc<RwLock<Vec<String>>>,
    state: Arc<RwLock<ConnectionState>>,
}

impl WorkerDiscovery {
    pub fn new(nostr: NostrDiscovery, doh: DohDiscovery, state: Arc<RwLock<ConnectionState>>) -> Self {
        info!("Initializing WorkerDiscovery");

        Self {
            nostr,
            doh,
            workers: Arc::new(RwLock::new(Vec::new())),
            state,
        }
    }

    pub async fn start(&self) -> PhantomResult<()> {
        info!("Starting worker discovery");

        *self.state.write().await = ConnectionState::Connecting;

        let nostr = &self.nostr;
        let workers = Arc::clone(&self.workers);
        let state = Arc::clone(&self.state);

        self.nostr.start_listener(move |new_workers| {
            let workers = Arc::clone(&workers);
            let state = Arc::clone(&state);

            tokio::spawn(async move {
                let mut lock = workers.write().await;
                *lock = new_workers;
                *state.write().await = ConnectionState::Connected;
                debug!(count = lock.len(), "Updated worker list from Nostr listener");
            });
        }).await?;

        self.refresh().await?;

        info!("Worker discovery started successfully");
        Ok(())
    }

    pub async fn get_workers(&self) -> Vec<String> {
        self.workers.read().await.clone()
    }

    pub async fn refresh(&self) -> PhantomResult<()> {
        debug!("Refreshing worker discovery");

        let mut all_workers = Vec::new();

        match self.nostr.discover_workers().await {
            Ok(workers) => {
                debug!(count = workers.len(), "Discovered workers via Nostr");
                all_workers.extend(workers);
            }
            Err(e) => {
                warn!(error = %e, "Nostr discovery failed");
            }
        }

        match self.doh.query_workers().await {
            Ok(workers) => {
                debug!(count = workers.len(), "Discovered workers via DoH");
                all_workers.extend(workers);
            }
            Err(e) => {
                warn!(error = %e, "DoH discovery failed");
            }
        }

        all_workers.sort();
        all_workers.dedup();

        if all_workers.is_empty() {
            error!("No workers discovered from any source");
            *self.state.write().await = ConnectionState::Failed;
            return Err(PhantomError::SignalingError(
                "No workers discovered from any source".to_string(),
            ));
        }

        {
            let mut lock = self.workers.write().await;
            *lock = all_workers.clone();
        }

        *self.state.write().await = ConnectionState::Connected;
        info!(count = all_workers.len(), "Worker discovery refreshed");
        Ok(())
    }

    pub fn get_state(&self) -> Arc<RwLock<ConnectionState>> {
        Arc::clone(&self.state)
    }
}
