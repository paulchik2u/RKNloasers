use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::utils::PhantomResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalingConfig {
    pub nostr_relays: Vec<String>,
    pub nostr_pubkey: String,
    pub doh_endpoint: String,
    pub doh_domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    pub protocol: String,
    pub alpn: Vec<String>,
    pub mtu: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskingConfig {
    pub protocol: String,
    pub gaming_profile: Option<GamingProfile>,
    pub heartbeat_interval_ms: u64,
    pub jitter_ms: u64,
    pub packet_size_min: u16,
    pub packet_size_max: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingProfile {
    pub name: String,
    pub packet_size_min: u16,
    pub packet_size_max: u16,
    pub heartbeat_interval_ms: u64,
    pub jitter_ms: u64,
    pub tick_rate_hz: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub algorithm: String,
    pub key_rotation_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitNodeConfig {
    pub region: String,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomConfig {
    pub signaling: SignalingConfig,
    pub transport: TransportConfig,
    pub masking: MaskingConfig,
    pub encryption: EncryptionConfig,
    pub exit_nodes: Vec<ExitNodeConfig>,
}

impl PhantomConfig {
    pub fn load_from_file(path: &str) -> PhantomResult<Self> {
        let path = Path::new(path);
        let content = std::fs::read_to_string(path)?;
        let config: PhantomConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}

impl Default for PhantomConfig {
    fn default() -> Self {
        Self {
            signaling: SignalingConfig {
                nostr_relays: vec![
                    "wss://relay.damus.io".to_string(),
                    "wss://relay.snort.social".to_string(),
                    "wss://nos.lol".to_string(),
                ],
                nostr_pubkey: String::new(),
                doh_endpoint: "https://cloudflare-dns.com/dns-query".to_string(),
                doh_domain: "cloudflare-dns.com".to_string(),
            },
            transport: TransportConfig {
                protocol: "quic".to_string(),
                alpn: vec!["h3".to_string()],
                mtu: 1200,
            },
            masking: MaskingConfig {
                protocol: "gaming".to_string(),
                gaming_profile: None,
                heartbeat_interval_ms: 30000,
                jitter_ms: 50,
                packet_size_min: 60,
                packet_size_max: 1400,
            },
            encryption: EncryptionConfig {
                algorithm: "chacha20-poly1305".to_string(),
                key_rotation_minutes: 60,
            },
            exit_nodes: vec![
                ExitNodeConfig {
                    region: "us-east".to_string(),
                    provider: "cloudflare".to_string(),
                },
                ExitNodeConfig {
                    region: "eu-west".to_string(),
                    provider: "cloudflare".to_string(),
                },
            ],
        }
    }
}
