use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, error, info};

use crate::utils::error::{PhantomError, PhantomResult};

#[derive(Debug, Deserialize)]
struct DohResponse {
    #[serde(rename = "Answer")]
    answer: Option<Vec<DnsRecord>>,
    #[serde(rename = "Status")]
    status: u16,
}

#[derive(Debug, Deserialize)]
struct DnsRecord {
    #[serde(rename = "data")]
    data: Option<String>,
    #[serde(rename = "type")]
    record_type: u16,
}

pub struct DohDiscovery {
    endpoint: String,
    domain: String,
    client: Client,
}

impl DohDiscovery {
    pub fn new(endpoint: String, domain: String) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        info!(
            endpoint = %endpoint,
            domain = %domain,
            "Initialized DoH discovery"
        );

        Self {
            endpoint,
            domain,
            client,
        }
    }

    pub async fn query_workers(&self) -> PhantomResult<Vec<String>> {
        debug!(domain = %self.domain, "Querying workers via DoH");

        let url = format!(
            "{}/dns-query?name={}&type=TXT",
            self.endpoint.trim_end_matches('/'),
            self.domain
        );

        let response = self
            .client
            .get(&url)
            .header("Accept", "application/dns-json")
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "DoH request failed");
                PhantomError::SignalingError(format!("DoH request failed: {e}"))
            })?;

        let status = response.status();
        if !status.is_success() {
            error!(status = %status, "DoH request returned non-success status");
            return Err(PhantomError::SignalingError(
                format!("DoH request returned status {status}"),
            ));
        }

        let body: DohResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse DoH response");
            PhantomError::SignalingError(format!("Failed to parse DoH response: {e}"))
        })?;

        if body.status != 0 {
            error!(status = body.status, "DNS query returned error status");
            return Err(PhantomError::SignalingError(format!(
                "DNS query returned error status: {}",
                body.status
            )));
        }

        let workers = body
            .answer
            .unwrap_or_default()
            .into_iter()
            .filter(|record| record.record_type == 16)
            .filter_map(|record| record.data)
            .filter_map(|data| {
                let trimmed = data.trim_matches('"');
                if trimmed.starts_with("phantom-worker=") {
                    Some(trimmed.strip_prefix("phantom-worker=").unwrap().to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>();

        debug!(count = workers.len(), "Discovered workers via DoH");
        Ok(workers)
    }
}
