use nostr_sdk::prelude::*;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::utils::error::{PhantomError, PhantomResult};

const WORKER_LIST_KIND: u64 = 30078;
const WORKER_LIST_IDENTIFIER: &str = "phantom-workers";

pub struct NostrDiscovery {
    relays: Vec<String>,
    client: Client,
    pubkey: String,
}

impl NostrDiscovery {
    pub fn new(relays: Vec<String>, pubkey: String) -> PhantomResult<Self> {
        info!(relays = ?relays, pubkey = %pubkey, "Initializing Nostr discovery");

        let keys = Keys::generate();
        let client = Client::new(keys);

        Ok(Self {
            relays,
            client,
            pubkey,
        })
    }

    async fn ensure_connected(&self) -> PhantomResult<()> {
        for relay in &self.relays {
            if let Err(e) = self.client.add_relay(relay).await {
                warn!(relay = %relay, error = %e, "Failed to add relay");
            }
        }

        self.client.connect().await;
        debug!("Connected to Nostr relays");
        Ok(())
    }

    pub async fn publish_workers(&self, workers: &[String]) -> PhantomResult<()> {
        info!(count = workers.len(), "Publishing worker list via Nostr");

        self.ensure_connected().await?;

        let content = serde_json::to_string(workers).map_err(|e| {
            error!(error = %e, "Failed to serialize worker list");
            PhantomError::SignalingError(format!("Serialization failed: {e}"))
        })?;

        let builder = EventBuilder::new(Kind::Custom(WORKER_LIST_KIND as u16), &content)
            .custom_tag(SingleLetterTag::lowercase(Alphabet::D), [WORKER_LIST_IDENTIFIER]);

        self.client.send_event_builder(builder).await.map_err(|e| {
            error!(error = %e, "Failed to publish worker list");
            PhantomError::SignalingError(format!("Failed to publish workers: {e}"))
        })?;

        info!("Worker list published successfully");
        Ok(())
    }

    pub async fn discover_workers(&self) -> PhantomResult<Vec<String>> {
        debug!("Discovering workers via Nostr");

        self.ensure_connected().await?;

        let pubkey = PublicKey::from_bech32(&self.pubkey).map_err(|e| {
            error!(error = %e, "Invalid pubkey");
            PhantomError::SignalingError(format!("Invalid pubkey: {e}"))
        })?;

        let filter = Filter::new()
            .author(pubkey)
            .kind(Kind::Custom(WORKER_LIST_KIND as u16))
            .limit(1);

        let events = self.client.fetch_events(vec![filter]).await.map_err(|e| {
            error!(error = %e, "Failed to fetch worker events");
            PhantomError::SignalingError(format!("Failed to fetch events: {e}"))
        })?;

        let mut workers = Vec::new();

        for event in events.iter() {
            for tag in event.tags.iter() {
                if tag.kind() == TagKind::SingleLetter(SingleLetterTag::lowercase(Alphabet::D)) {
                    if tag.content() == Some(WORKER_LIST_IDENTIFIER) {
                        match serde_json::from_str::<Vec<String>>(&event.content) {
                            Ok(parsed) => {
                                debug!(count = parsed.len(), "Parsed worker list from event");
                                workers.extend(parsed);
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to parse worker list from event");
                            }
                        }
                    }
                }
            }
        }

        debug!(count = workers.len(), "Discovered workers via Nostr");
        Ok(workers)
    }

    pub async fn start_listener(
        &self,
        callback: impl Fn(Vec<String>) + Send + Sync + 'static,
    ) -> PhantomResult<()> {
        info!("Starting Nostr worker listener");

        self.ensure_connected().await?;

        let pubkey = PublicKey::from_bech32(&self.pubkey).map_err(|e| {
            error!(error = %e, "Invalid pubkey");
            PhantomError::SignalingError(format!("Invalid pubkey: {e}"))
        })?;

        let filter = Filter::new()
            .author(pubkey)
            .kind(Kind::Custom(WORKER_LIST_KIND as u16));

        self.client.subscribe(vec![filter], None).await;

        let client = self.client.clone();
        let callback = Arc::new(callback);

        tokio::spawn(async move {
            let mut notifications = client.notifications();

            while let Ok(notification) = notifications.recv().await {
                if let RelayPoolNotification::Event { event, .. } = notification {
                    if event.kind == Kind::Custom(WORKER_LIST_KIND as u16) {
                        match serde_json::from_str::<Vec<String>>(&event.content) {
                            Ok(workers) => {
                                debug!(count = workers.len(), "Received worker update via Nostr");
                                callback(workers);
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to parse worker update");
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }
}
