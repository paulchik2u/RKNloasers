use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

use crate::utils::{PhantomError, PhantomResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianComponentDef {
    pub mean: f64,
    pub std_dev: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBinDef {
    pub time_ms: f64,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstPatternDef {
    pub name: String,
    pub packet_count_range: (usize, usize),
    pub inter_packet_time_range: (f64, f64),
    pub size_range: (usize, usize),
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyProfileDef {
    pub mean_entropy: f64,
    pub entropy_std: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingProfile {
    pub name: String,
    pub packet_size_min: u16,
    pub packet_size_max: u16,
    pub heartbeat_interval_ms: u64,
    pub jitter_ms: u64,
    pub frequency_hz: u16,
    #[serde(default)]
    pub size_distribution: Vec<GaussianComponentDef>,
    #[serde(default)]
    pub timing_distribution: Vec<TimingBinDef>,
    #[serde(default)]
    pub burst_patterns: Vec<BurstPatternDef>,
    #[serde(default = "default_direction_ratio")]
    pub client_server_size_ratio: (f64, f64),
    #[serde(default)]
    pub entropy_profile: Option<EntropyProfileDef>,
}

fn default_direction_ratio() -> (f64, f64) {
    (0.6, 1.0)
}

impl GamingProfile {
    pub fn load_from_file(path: &str) -> PhantomResult<Self> {
        tracing::debug!(path = %path, "Loading gaming profile from file");
        let content = fs::read_to_string(path).map_err(|e| {
            tracing::error!(path = %path, error = %e, "Failed to read profile file");
            PhantomError::ConfigError(format!("Failed to read profile file {}: {}", path, e))
        })?;
        let profile: GamingProfile = serde_json::from_str(&content).map_err(|e| {
            tracing::error!(path = %path, error = %e, "Failed to parse profile JSON");
            PhantomError::ConfigError(format!("Failed to parse profile {}: {}", path, e))
        })?;
        tracing::info!(name = %profile.name, "Loaded gaming profile");
        Ok(profile)
    }

    pub fn default_profiles() -> HashMap<String, GamingProfile> {
        let profile_names = ["valorant", "cs2", "minecraft"];
        let mut profiles = HashMap::with_capacity(profile_names.len());

        for name in &profile_names {
            let path = format!("config/gaming-profiles/{}.json", name);
            match GamingProfile::load_from_file(&path) {
                Ok(profile) => {
                    profiles.insert(name.to_string(), profile);
                }
                Err(e) => {
                    tracing::warn!(profile = %name, error = %e, "Failed to load default profile, using fallback");
                    let fallback = fallback_profile(name);
                    profiles.insert(name.to_string(), fallback);
                }
            }
        }

        profiles
    }
}

impl Default for GamingProfile {
    fn default() -> Self {
        GamingProfile {
            name: "valorant".to_string(),
            packet_size_min: 60,
            packet_size_max: 300,
            heartbeat_interval_ms: 30000,
            jitter_ms: 50,
            frequency_hz: 60,
        }
    }
}

fn fallback_profile(name: &str) -> GamingProfile {
    match name {
        "valorant" => GamingProfile {
            name: "valorant".to_string(),
            packet_size_min: 60,
            packet_size_max: 300,
            heartbeat_interval_ms: 30000,
            jitter_ms: 50,
            frequency_hz: 60,
            size_distribution: vec![
                GaussianComponentDef { mean: 75.0, std_dev: 12.0, weight: 0.55 },
                GaussianComponentDef { mean: 180.0, std_dev: 40.0, weight: 0.30 },
                GaussianComponentDef { mean: 280.0, std_dev: 20.0, weight: 0.15 },
            ],
            timing_distribution: vec![
                TimingBinDef { time_ms: 16.67, probability: 0.70 },
                TimingBinDef { time_ms: 33.33, probability: 0.15 },
                TimingBinDef { time_ms: 50.0, probability: 0.10 },
                TimingBinDef { time_ms: 100.0, probability: 0.05 },
            ],
            burst_patterns: vec![
                BurstPatternDef {
                    name: "shooting".to_string(),
                    packet_count_range: (5, 10),
                    inter_packet_time_range: (5.0, 15.0),
                    size_range: (60, 150),
                    probability: 0.12,
                },
                BurstPatternDef {
                    name: "movement".to_string(),
                    packet_count_range: (2, 4),
                    inter_packet_time_range: (15.0, 20.0),
                    size_range: (60, 100),
                    probability: 0.40,
                },
            ],
            client_server_size_ratio: (0.55, 1.0),
            entropy_profile: Some(EntropyProfileDef { mean_entropy: 7.2, entropy_std: 0.3 }),
        },
        "cs2" => GamingProfile {
            name: "cs2".to_string(),
            packet_size_min: 80,
            packet_size_max: 400,
            heartbeat_interval_ms: 25000,
            jitter_ms: 40,
            frequency_hz: 64,
            size_distribution: vec![
                GaussianComponentDef { mean: 90.0, std_dev: 15.0, weight: 0.50 },
                GaussianComponentDef { mean: 200.0, std_dev: 50.0, weight: 0.30 },
                GaussianComponentDef { mean: 400.0, std_dev: 80.0, weight: 0.15 },
                GaussianComponentDef { mean: 1000.0, std_dev: 100.0, weight: 0.05 },
            ],
            timing_distribution: vec![
                TimingBinDef { time_ms: 15.63, probability: 0.65 },
                TimingBinDef { time_ms: 31.25, probability: 0.18 },
                TimingBinDef { time_ms: 46.88, probability: 0.10 },
                TimingBinDef { time_ms: 100.0, probability: 0.07 },
            ],
            burst_patterns: vec![
                BurstPatternDef {
                    name: "shooting".to_string(),
                    packet_count_range: (5, 10),
                    inter_packet_time_range: (5.0, 12.0),
                    size_range: (80, 200),
                    probability: 0.15,
                },
                BurstPatternDef {
                    name: "movement".to_string(),
                    packet_count_range: (2, 5),
                    inter_packet_time_range: (14.0, 18.0),
                    size_range: (80, 120),
                    probability: 0.45,
                },
                BurstPatternDef {
                    name: "map_load".to_string(),
                    packet_count_range: (15, 30),
                    inter_packet_time_range: (2.0, 8.0),
                    size_range: (800, 1200),
                    probability: 0.02,
                },
            ],
            client_server_size_ratio: (0.50, 1.0),
            entropy_profile: Some(EntropyProfileDef { mean_entropy: 7.4, entropy_std: 0.2 }),
        },
        "minecraft" => GamingProfile {
            name: "minecraft".to_string(),
            packet_size_min: 100,
            packet_size_max: 1400,
            heartbeat_interval_ms: 20000,
            jitter_ms: 100,
            frequency_hz: 20,
            size_distribution: vec![
                GaussianComponentDef { mean: 150.0, std_dev: 40.0, weight: 0.40 },
                GaussianComponentDef { mean: 350.0, std_dev: 80.0, weight: 0.35 },
                GaussianComponentDef { mean: 700.0, std_dev: 150.0, weight: 0.20 },
                GaussianComponentDef { mean: 1200.0, std_dev: 100.0, weight: 0.05 },
            ],
            timing_distribution: vec![
                TimingBinDef { time_ms: 50.0, probability: 0.50 },
                TimingBinDef { time_ms: 100.0, probability: 0.25 },
                TimingBinDef { time_ms: 200.0, probability: 0.15 },
                TimingBinDef { time_ms: 500.0, probability: 0.10 },
            ],
            burst_patterns: vec![
                BurstPatternDef {
                    name: "chunk_load".to_string(),
                    packet_count_range: (10, 25),
                    inter_packet_time_range: (5.0, 20.0),
                    size_range: (500, 1400),
                    probability: 0.05,
                },
                BurstPatternDef {
                    name: "entity_update".to_string(),
                    packet_count_range: (3, 8),
                    inter_packet_time_range: (40.0, 60.0),
                    size_range: (100, 400),
                    probability: 0.25,
                },
            ],
            client_server_size_ratio: (0.45, 1.0),
            entropy_profile: Some(EntropyProfileDef { mean_entropy: 6.8, entropy_std: 0.4 }),
        },
        _ => GamingProfile::default(),
    }
}
