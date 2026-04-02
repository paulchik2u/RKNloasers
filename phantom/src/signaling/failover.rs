use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::utils::error::{PhantomError, PhantomResult};
use crate::signaling::discovery::WorkerDiscovery;

pub struct FailoverManager {
    discovery: Arc<WorkerDiscovery>,
    active_worker: Arc<RwLock<String>>,
    health_check_interval: Duration,
}

impl FailoverManager {
    pub fn new(discovery: Arc<WorkerDiscovery>) -> Self {
        info!("Initializing FailoverManager");

        Self {
            discovery,
            active_worker: Arc::new(RwLock::new(String::new())),
            health_check_interval: Duration::from_secs(30),
        }
    }

    pub async fn start(&self) -> PhantomResult<()> {
        info!("Starting FailoverManager");

        self.switch_worker().await?;

        let discovery = Arc::clone(&self.discovery);
        let active_worker = Arc::clone(&self.active_worker);
        let interval = self.health_check_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                ticker.tick().await;

                let current_worker = active_worker.read().await.clone();
                if !Self::check_health_internal(&current_worker).await {
                    warn!(worker = %current_worker, "Health check failed, switching worker");

                    let workers = discovery.get_workers().await;
                    if workers.is_empty() {
                        error!("No workers available for failover");
                        continue;
                    }

                    let mut best_worker = None;
                    for worker in &workers {
                        if worker != &current_worker && Self::check_health_internal(worker).await {
                            best_worker = Some(worker.clone());
                            break;
                        }
                    }

                    if let Some(healthy_worker) = best_worker {
                        info!(worker = %healthy_worker, "Switched to healthy worker");
                        *active_worker.write().await = healthy_worker;
                    } else {
                        error!("No healthy workers available for failover");
                    }
                }
            }
        });

        Ok(())
    }

    pub async fn check_health(&self, worker_url: &str) -> bool {
        Self::check_health_internal(worker_url).await
    }

    async fn check_health_internal(worker_url: &str) -> bool {
        if worker_url.is_empty() {
            return false;
        }

        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "Failed to build health check client");
                return false;
            }
        };

        match client.get(worker_url).send().await {
            Ok(response) => {
                let healthy = response.status().is_success();
                debug!(url = %worker_url, healthy, "Health check result");
                healthy
            }
            Err(e) => {
                debug!(url = %worker_url, error = %e, "Health check failed");
                false
            }
        }
    }

    pub async fn switch_worker(&self) -> PhantomResult<()> {
        debug!("Switching to best available worker");

        let workers = self.discovery.get_workers().await;
        if workers.is_empty() {
            error!("No workers available to switch to");
            return Err(PhantomError::SignalingError(
                "No workers available for failover".to_string(),
            ));
        }

        for worker in &workers {
            if Self::check_health_internal(worker).await {
                info!(worker = %worker, "Selected new active worker");
                *self.active_worker.write().await = worker.clone();
                return Ok(());
            }
        }

        warn!("No healthy workers found, selecting first available");
        *self.active_worker.write().await = workers[0].clone();
        Ok(())
    }

    pub async fn get_active_worker(&self) -> String {
        self.active_worker.read().await.clone()
    }
}
