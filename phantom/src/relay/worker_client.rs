use std::sync::Arc;
use tokio::sync::RwLock;

use reqwest::Client;

use crate::phantom_core::state::ConnectionState;
use crate::utils::error::{PhantomError, PhantomResult};

const PHANTOM_VERSION: &str = "0.1.0";
const REQUEST_TIMEOUT_SECS: u64 = 30;

pub struct WorkerClient {
    worker_url: String,
    client: Client,
    state: Arc<RwLock<ConnectionState>>,
}

impl WorkerClient {
    pub fn new(worker_url: String, state: Arc<RwLock<ConnectionState>>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .http3_prior_knowledge()
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            worker_url,
            client,
            state,
        }
    }

    pub fn url(&self) -> &str {
        &self.worker_url
    }

    pub async fn send(&self, data: &[u8]) -> PhantomResult<Vec<u8>> {
        tracing::debug!(
            url = %self.worker_url,
            bytes = data.len(),
            "Sending data to worker"
        );

        let response = self
            .client
            .post(&self.worker_url)
            .header("x-phantom-version", PHANTOM_VERSION)
            .header("content-type", "application/octet-stream")
            .body(data.to_vec())
            .send()
            .await
            .map_err(|e| {
                tracing::error!(
                    url = %self.worker_url,
                    error = %e,
                    "Failed to send request to worker"
                );
                PhantomError::RelayError(format!("Worker request failed: {e}"))
            })?;

        let status = response.status();

        if !status.is_success() {
            let status_code = status.as_u16();
            tracing::warn!(
                url = %self.worker_url,
                status = status_code,
                "Worker returned error status"
            );
            return Err(PhantomError::RelayError(format!(
                "Worker returned status {status_code}"
            )));
        }

        let body = response.bytes().await.map_err(|e| {
            tracing::error!(
                url = %self.worker_url,
                error = %e,
                "Failed to read response body from worker"
            );
            PhantomError::RelayError(format!("Failed to read response body: {e}"))
        })?;

        tracing::debug!(
            url = %self.worker_url,
            bytes = body.len(),
            "Received response from worker"
        );

        Ok(body.to_vec())
    }
}
