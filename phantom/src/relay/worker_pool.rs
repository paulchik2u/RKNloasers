use std::sync::Arc;
use tokio::sync::RwLock;

use crate::phantom_core::state::ConnectionState;
use crate::relay::worker_client::WorkerClient;
use crate::utils::error::{PhantomError, PhantomResult};

pub struct WorkerPool {
    workers: Arc<RwLock<Vec<WorkerClient>>>,
    active_index: Arc<RwLock<usize>>,
    state: Arc<RwLock<ConnectionState>>,
}

impl WorkerPool {
    pub fn new(worker_urls: Vec<String>, state: Arc<RwLock<ConnectionState>>) -> Self {
        let workers: Vec<WorkerClient> = worker_urls
            .into_iter()
            .map(|url| WorkerClient::new(url, Arc::clone(&state)))
            .collect();

        Self {
            workers: Arc::new(RwLock::new(workers)),
            active_index: Arc::new(RwLock::new(0)),
            state,
        }
    }

    pub async fn send(&self, data: &[u8]) -> PhantomResult<Vec<u8>> {
        let workers = self.workers.read().await;
        let worker_count = workers.len();

        if worker_count == 0 {
            return Err(PhantomError::RelayError(
                "No workers available in pool".to_string(),
            ));
        }

        let start_index = *self.active_index.read().await;

        for offset in 0..worker_count {
            let index = (start_index + offset) % worker_count;

            if index >= workers.len() {
                continue;
            }

            let worker = &workers[index];
            let url = worker.url().to_string();

            match worker.send(data).await {
                Ok(response) => {
                    let mut active = self.active_index.write().await;
                    *active = index;
                    tracing::debug!(
                        url = %url,
                        index = index,
                        "Successfully sent via worker"
                    );
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!(
                        url = %url,
                        index = index,
                        error = %e,
                        "Worker failed, trying next"
                    );
                    continue;
                }
            }
        }

        tracing::error!("All workers in pool failed");
        Err(PhantomError::RelayError(
            "All workers in pool failed".to_string(),
        ))
    }

    pub async fn add_worker(&self, url: String) {
        tracing::info!(url = %url, "Adding worker to pool");
        let worker = WorkerClient::new(url.clone(), Arc::clone(&self.state));
        let mut workers = self.workers.write().await;
        workers.push(worker);
        tracing::info!(
            url = %url,
            count = workers.len(),
            "Worker added to pool"
        );
    }

    pub async fn remove_worker(&self, url: &str) {
        tracing::info!(url = %url, "Removing worker from pool");
        let mut workers = self.workers.write().await;
        let initial_len = workers.len();
        workers.retain(|w| w.url() != url);
        let removed = initial_len - workers.len();

        if removed > 0 {
            let mut active = self.active_index.write().await;
            if *active >= workers.len() && !workers.is_empty() {
                *active = workers.len() - 1;
            }
            tracing::info!(
                url = %url,
                count = workers.len(),
                "Worker removed from pool"
            );
        } else {
            tracing::warn!(url = %url, "Worker not found in pool");
        }
    }

    pub async fn get_active_worker(&self) -> String {
        let workers = self.workers.read().await;
        let index = *self.active_index.read().await;

        if workers.is_empty() {
            return String::new();
        }

        let safe_index = index.min(workers.len() - 1);
        workers[safe_index].url().to_string()
    }

    pub async fn worker_count(&self) -> usize {
        self.workers.read().await.len()
    }
}
