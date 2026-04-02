use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

use crate::masking::statistical_model::GaussianComponent;
use crate::utils::{PhantomError, PhantomResult};

#[derive(Debug, Clone)]
pub enum EncryptionType {
    None,
    Tls,
    Dtls,
    Custom,
    Unknown,
}

impl std::fmt::Display for EncryptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptionType::None => write!(f, "none"),
            EncryptionType::Tls => write!(f, "tls"),
            EncryptionType::Dtls => write!(f, "dtls"),
            EncryptionType::Custom => write!(f, "custom"),
            EncryptionType::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DirectionPattern {
    Symmetric,
    ClientHeavy,
    ServerHeavy,
    Bursty,
}

impl std::fmt::Display for DirectionPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectionPattern::Symmetric => write!(f, "symmetric"),
            DirectionPattern::ClientHeavy => write!(f, "client_heavy"),
            DirectionPattern::ServerHeavy => write!(f, "server_heavy"),
            DirectionPattern::Bursty => write!(f, "bursty"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProtocolSignature {
    pub header_pattern: Vec<u8>,
    pub typical_ports: Vec<u16>,
    pub encryption_type: EncryptionType,
    pub direction_pattern: DirectionPattern,
}

#[derive(Debug, Clone)]
pub struct ProtocolProfile {
    pub name: String,
    pub packet_size_dist: Vec<GaussianComponent>,
    pub timing_dist: Vec<(f64, f64)>,
    pub port_range: (u16, u16),
    pub protocol_signature: ProtocolSignature,
}

#[derive(Debug, Clone)]
pub struct ProtocolStatistics {
    pub mean_packet_size: f64,
    pub std_packet_size: f64,
    pub mean_inter_packet_time: f64,
    pub bytes_up: u64,
    pub bytes_down: u64,
    pub packet_count: u64,
}

impl ProtocolStatistics {
    fn new() -> Self {
        Self {
            mean_packet_size: 0.0,
            std_packet_size: 0.0,
            mean_inter_packet_time: 0.0,
            bytes_up: 0,
            bytes_down: 0,
            packet_count: 0,
        }
    }

    fn update(&mut self, packet_size: usize, inter_packet_time: f64, is_upload: bool) {
        self.packet_count += 1;
        let count = self.packet_count as f64;
        let size = packet_size as f64;

        let old_mean = self.mean_packet_size;
        self.mean_packet_size += (size - old_mean) / count;

        if self.packet_count > 1 {
            let m2 = self.std_packet_size * self.std_packet_size * (count - 1.0);
            let new_m2 = m2 + (size - old_mean) * (size - self.mean_packet_size);
            self.std_packet_size = (new_m2 / (count - 1.0)).sqrt();
        }

        let old_ipt = self.mean_inter_packet_time;
        self.mean_inter_packet_time += (inter_packet_time - old_ipt) / count;

        if is_upload {
            self.bytes_up += packet_size as u64;
        } else {
            self.bytes_down += packet_size as u64;
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlendedPacket {
    pub data: Vec<u8>,
    pub protocol_hint: String,
    pub delay: Duration,
    pub size_adjustment: usize,
}

pub struct ProtocolBlender {
    profiles: Vec<ProtocolProfile>,
    blend_weights: Vec<f64>,
    current_protocol: usize,
    switch_counter: usize,
    protocol_stats: Vec<ProtocolStatistics>,
    timing_cdfs: Vec<Vec<f64>>,
    last_switch_weight: f64,
    consecutive_same_protocol: usize,
}

impl ProtocolBlender {
    pub fn new(profiles: Vec<ProtocolProfile>) -> PhantomResult<Self> {
        if profiles.len() < 2 {
            return Err(PhantomError::ConfigError(
                "ProtocolBlender requires at least 2 profiles".to_string(),
            ));
        }

        for profile in &profiles {
            if profile.packet_size_dist.is_empty() {
                return Err(PhantomError::ConfigError(format!(
                    "Profile '{}' has empty packet size distribution",
                    profile.name
                )));
            }
            if profile.timing_dist.is_empty() {
                return Err(PhantomError::ConfigError(format!(
                    "Profile '{}' has empty timing distribution",
                    profile.name
                )));
            }
            let total_weight: f64 = profile.packet_size_dist.iter().map(|c| c.weight).sum();
            if total_weight <= 0.0 {
                return Err(PhantomError::ConfigError(format!(
                    "Profile '{}' has zero total weight in size distribution",
                    profile.name
                )));
            }
        }

        let n = profiles.len();
        let blend_weights = vec![1.0 / n as f64; n];
        let protocol_stats = vec![ProtocolStatistics::new(); n];

        let timing_cdfs: Vec<Vec<f64>> = profiles
            .iter()
            .map(|p| {
                let probs: Vec<f64> = p.timing_dist.iter().map(|(_, prob)| *prob).collect();
                Self::compute_cdf(&probs)
            })
            .collect();

        tracing::info!(
            profile_count = n,
            profiles = ?profiles.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            "ProtocolBlender initialized"
        );

        Ok(Self {
            profiles,
            blend_weights,
            current_protocol: 0,
            switch_counter: 0,
            protocol_stats,
            timing_cdfs,
            last_switch_weight: 1.0 / n as f64,
            consecutive_same_protocol: 0,
        })
    }

    pub fn select_protocol(&mut self, rng: &mut impl Rng) -> usize {
        let selected = self.sample_weighted_index(rng);

        if selected != self.current_protocol {
            self.current_protocol = selected;
            self.switch_counter += 1;
            self.consecutive_same_protocol = 0;
            self.last_switch_weight = self.blend_weights[selected];

            tracing::trace!(
                to_protocol = %self.profiles[selected].name,
                switch_count = self.switch_counter,
                weight = self.last_switch_weight,
                "Protocol switch"
            );
        } else {
            self.consecutive_same_protocol += 1;
        }

        selected
    }

    pub fn adapt_weights(&mut self, detection_risk: f64) {
        let risk = detection_risk.clamp(0.0, 1.0);
        let n = self.profiles.len() as f64;
        let base_weight = 1.0 / n;

        let smoothing_factor = if risk < 0.3 {
            0.05
        } else if risk < 0.6 {
            0.15
        } else {
            0.30
        };

        let entropy = self.blend_entropy();
        let max_entropy = n.ln();
        let entropy_ratio = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            1.0
        };

        let diversity_pressure = (1.0 - entropy_ratio) * 0.2;

        let mut new_weights = Vec::with_capacity(self.profiles.len());

        for (i, _profile) in self.profiles.iter().enumerate() {
            let current = self.blend_weights[i];

            let risk_adjustment = if i == self.current_protocol {
                -risk * smoothing_factor
            } else {
                risk * smoothing_factor / (n - 1.0)
            };

            let diversity_boost = diversity_pressure * (base_weight - current);

            let profile_inertia = self.compute_profile_inertia(i);
            let inertia_factor = 1.0 - profile_inertia * smoothing_factor * 0.5;

            let mut target = current + risk_adjustment + diversity_boost;
            target *= inertia_factor;

            let min_weight = base_weight * 0.1;
            let max_weight = base_weight * 3.0;
            target = target.clamp(min_weight, max_weight);

            new_weights.push(target);
        }

        let total: f64 = new_weights.iter().sum();
        if total > 0.0 {
            for w in &mut new_weights {
                *w /= total;
            }
        } else {
            for w in &mut new_weights {
                *w = base_weight;
            }
        }

        let alpha = smoothing_factor.min(0.5);
        for i in 0..self.blend_weights.len() {
            self.blend_weights[i] = (1.0 - alpha) * self.blend_weights[i] + alpha * new_weights[i];
        }

        let final_total: f64 = self.blend_weights.iter().sum();
        if final_total > 0.0 {
            for w in &mut self.blend_weights {
                *w /= final_total;
            }
        }

        tracing::debug!(
            detection_risk = risk,
            entropy = entropy,
            smoothing = smoothing_factor,
            weights = ?self.blend_weights.iter().map(|w| (*w * 100.0).round() / 100.0).collect::<Vec<_>>(),
            "Blend weights adapted"
        );
    }

    pub fn current_profile(&self) -> &ProtocolProfile {
        &self.profiles[self.current_protocol]
    }

    pub fn blend_packet(&mut self, data: Vec<u8>, rng: &mut impl Rng) -> PhantomResult<BlendedPacket> {
        let profile_idx = self.current_protocol;
        let profile = &self.profiles[profile_idx];

        let target_size = self.sample_packet_size(profile_idx, rng);
        let ipt = self.sample_inter_packet_time(profile_idx, rng);

        if target_size > data.len() {
            let padding = target_size.saturating_sub(data.len());
            let mut padded = Vec::with_capacity(target_size);
            padded.extend_from_slice(&data);
            padded.extend(self.generate_padding(padding, profile, rng));

            let delay = Duration::from_secs_f64(ipt / 1000.0);
            let is_upload = rng.gen::<f64>() < self.direction_upload_probability(profile);

            self.protocol_stats[profile_idx].update(target_size, ipt, is_upload);

            tracing::trace!(
                profile = %profile.name,
                original_size = data.len(),
                target_size = target_size,
                padding = padding,
                delay_ms = ipt,
                "Blended packet generated (padded)"
            );

            return Ok(BlendedPacket {
                data: padded,
                protocol_hint: profile.name.clone(),
                delay,
                size_adjustment: padding,
            });
        } else if target_size < data.len() {
            let chunk_size = target_size.max(1);
            let header_size = profile.protocol_signature.header_pattern.len();
            let payload_size = chunk_size.saturating_sub(header_size).max(1);

            let mut output = Vec::with_capacity(chunk_size);
            output.extend_from_slice(&profile.protocol_signature.header_pattern);
            let data_end = payload_size.min(data.len());
            output.extend_from_slice(&data[..data_end]);

            let delay = Duration::from_secs_f64(ipt / 1000.0);
            let is_upload = rng.gen::<f64>() < self.direction_upload_probability(profile);

            self.protocol_stats[profile_idx].update(output.len(), ipt, is_upload);

            tracing::trace!(
                profile = %profile.name,
                original_size = data.len(),
                target_size = target_size,
                output_size = output.len(),
                delay_ms = ipt,
                "Blended packet generated (truncated)"
            );

            return Ok(BlendedPacket {
                data: output,
                protocol_hint: profile.name.clone(),
                delay,
                size_adjustment: 0,
            });
        }

        let delay = Duration::from_secs_f64(ipt / 1000.0);
        let is_upload = rng.gen::<f64>() < self.direction_upload_probability(profile);

        self.protocol_stats[profile_idx].update(data.len(), ipt, is_upload);

        tracing::trace!(
            profile = %profile.name,
            size = data.len(),
            delay_ms = ipt,
            "Blended packet generated (unchanged size)"
        );

        Ok(BlendedPacket {
            data,
            protocol_hint: profile.name.clone(),
            delay,
            size_adjustment: 0,
        })
    }

    pub fn blend_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for &w in &self.blend_weights {
            if w > 0.0 {
                entropy -= w * w.ln();
            }
        }
        entropy
    }

    pub fn profiles(&self) -> &[ProtocolProfile] {
        &self.profiles
    }

    pub fn blend_weights(&self) -> &[f64] {
        &self.blend_weights
    }

    pub fn switch_counter(&self) -> usize {
        self.switch_counter
    }

    pub fn protocol_stats(&self) -> &[ProtocolStatistics] {
        &self.protocol_stats
    }

    pub fn profile_by_name(&self, name: &str) -> Option<usize> {
        self.profiles.iter().position(|p| p.name == name)
    }

    fn sample_weighted_index(&self, rng: &mut impl Rng) -> usize {
        let roll: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (i, &weight) in self.blend_weights.iter().enumerate() {
            cumulative += weight;
            if roll < cumulative {
                return i;
            }
        }

        self.blend_weights.len() - 1
    }

    fn compute_cdf(weights: &[f64]) -> Vec<f64> {
        let mut cdf = Vec::with_capacity(weights.len());
        let mut cumulative = 0.0;

        for &w in weights {
            cumulative += w;
            cdf.push(cumulative);
        }

        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        cdf
    }

    fn sample_from_cdf(cdf: &[f64], rng: &mut impl Rng) -> usize {
        let roll: f64 = rng.gen();

        match cdf.binary_search_by(|&val| {
            if val < roll {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }) {
            Ok(idx) => idx,
            Err(idx) => idx.min(cdf.len().saturating_sub(1)),
        }
    }

    fn sample_packet_size(&self, profile_idx: usize, rng: &mut impl Rng) -> usize {
        let profile = &self.profiles[profile_idx];

        let weights: Vec<f64> = profile.packet_size_dist.iter().map(|c| c.weight).collect();
        let cdf = Self::compute_cdf(&weights);
        let component_idx = Self::sample_from_cdf(&cdf, rng);
        let component = &profile.packet_size_dist[component_idx];

        let normal = match Normal::new(component.mean, component.std_dev.max(1.0)) {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    profile = %profile.name,
                    mean = component.mean,
                    std = component.std_dev,
                    "Failed to create normal distribution for packet size"
                );
                return component.mean.round() as usize;
            }
        };

        let raw = normal.sample(rng);
        let port_min = profile.port_range.0 as f64;
        let clamped = raw.clamp(40.0, 1500.0).max(port_min.min(1500.0));
        clamped.round() as usize
    }

    fn sample_inter_packet_time(&self, profile_idx: usize, rng: &mut impl Rng) -> f64 {
        let profile = &self.profiles[profile_idx];
        let cdf = &self.timing_cdfs[profile_idx];

        if cdf.is_empty() {
            tracing::warn!(profile = %profile.name, "Empty timing CDF");
            return 50.0;
        }

        let idx = Self::sample_from_cdf(cdf, rng);
        let (base_time, _) = profile.timing_dist[idx];

        let jitter_std = base_time * 0.05;
        let jitter = match Normal::new(0.0, jitter_std.max(0.5)) {
            Ok(n) => n.sample(rng),
            Err(_) => 0.0,
        };

        (base_time + jitter).max(1.0)
    }

    fn generate_padding(&self, size: usize, profile: &ProtocolProfile, rng: &mut impl Rng) -> Vec<u8> {
        let mut padding = Vec::with_capacity(size);

        let entropy_target = self.compute_profile_entropy_target(profile);
        let byte_range = self.entropy_to_byte_range(entropy_target);

        for _ in 0..size {
            let byte = rng.gen_range(byte_range.0..=byte_range.1);
            padding.push(byte);
        }

        if !profile.protocol_signature.header_pattern.is_empty() {
            let header_len = profile.protocol_signature.header_pattern.len().min(size);
            if header_len > 0 && rng.gen::<f64>() < 0.3 {
                for (i, &byte) in profile.protocol_signature.header_pattern.iter().take(header_len).enumerate() {
                    padding[i] = byte;
                }
            }
        }

        padding
    }

    fn compute_profile_entropy_target(&self, profile: &ProtocolProfile) -> f64 {
        let mut entropy = 0.0;
        let total_weight: f64 = profile.packet_size_dist.iter().map(|c| c.weight).sum();

        for component in &profile.packet_size_dist {
            let w = component.weight / total_weight.max(f64::MIN_POSITIVE);
            if w > 0.0 {
                let component_entropy = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * component.std_dev * component.std_dev).ln();
                entropy += w * component_entropy;
            }
        }

        entropy.clamp(0.0, 8.0)
    }

    fn entropy_to_byte_range(&self, entropy: f64) -> (u8, u8) {
        let ratio = (entropy / 8.0).clamp(0.0, 1.0);
        let spread = (ratio * 255.0) as u8;

        if spread < 64 {
            (96, 96u8.saturating_add(spread))
        } else if spread < 128 {
            (64, 64u8.saturating_add(spread))
        } else if spread < 192 {
            (32, 32u8.saturating_add(spread))
        } else {
            (0, 255)
        }
    }

    fn direction_upload_probability(&self, profile: &ProtocolProfile) -> f64 {
        match profile.protocol_signature.direction_pattern {
            DirectionPattern::Symmetric => 0.5,
            DirectionPattern::ClientHeavy => 0.65,
            DirectionPattern::ServerHeavy => 0.35,
            DirectionPattern::Bursty => {
                let burst_phase = (self.switch_counter as f64 * 0.1).sin() * 0.5 + 0.5;
                0.3 + burst_phase * 0.4
            }
        }
    }

    fn compute_profile_inertia(&self, profile_idx: usize) -> f64 {
        let stats = &self.protocol_stats[profile_idx];
        if stats.packet_count < 10 {
            return 0.0;
        }

        let cv = if stats.mean_packet_size > 0.0 {
            stats.std_packet_size / stats.mean_packet_size
        } else {
            0.0
        };

        let stability = 1.0 - cv.clamp(0.0, 1.0);
        let volume_factor = (stats.packet_count as f64).ln() / 10.0;

        (stability * 0.6 + volume_factor.clamp(0.0, 1.0) * 0.4)
            .clamp(0.0, 1.0)
    }

    pub fn compute_blend_chi_squared(&self, observed_switches: &[usize]) -> f64 {
        if observed_switches.len() < 2 || self.profiles.len() < 2 {
            return f64::INFINITY;
        }

        let n = self.profiles.len();
        let total: usize = observed_switches.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mut chi_sq = 0.0;
        for i in 0..n {
            let expected = self.blend_weights[i] * total as f64;
            if expected > 0.0 {
                let diff = observed_switches[i] as f64 - expected;
                chi_sq += (diff * diff) / expected;
            }
        }

        chi_sq
    }

    pub fn reset_statistics(&mut self) {
        for stats in &mut self.protocol_stats {
            *stats = ProtocolStatistics::new();
        }
        self.switch_counter = 0;
        self.consecutive_same_protocol = 0;

        tracing::info!("Protocol blender statistics reset");
    }
}

pub fn create_default_profiles() -> PhantomResult<Vec<ProtocolProfile>> {
    let cs2 = ProtocolProfile {
        name: "cs2".to_string(),
        packet_size_dist: vec![
            GaussianComponent { mean: 90.0, std_dev: 15.0, weight: 0.50 },
            GaussianComponent { mean: 200.0, std_dev: 50.0, weight: 0.30 },
            GaussianComponent { mean: 400.0, std_dev: 80.0, weight: 0.15 },
            GaussianComponent { mean: 1000.0, std_dev: 100.0, weight: 0.05 },
        ],
        timing_dist: vec![
            (15.63, 0.65),
            (31.25, 0.18),
            (46.88, 0.10),
            (100.0, 0.07),
        ],
        port_range: (27000, 27100),
        protocol_signature: ProtocolSignature {
            header_pattern: vec![0xFF, 0xFF, 0xFF, 0xFF],
            typical_ports: vec![27015, 27016, 27017],
            encryption_type: EncryptionType::Dtls,
            direction_pattern: DirectionPattern::ClientHeavy,
        },
    };

    let discord_voice = ProtocolProfile {
        name: "discord_voice".to_string(),
        packet_size_dist: vec![
            GaussianComponent { mean: 160.0, std_dev: 30.0, weight: 0.60 },
            GaussianComponent { mean: 320.0, std_dev: 60.0, weight: 0.25 },
            GaussianComponent { mean: 640.0, std_dev: 100.0, weight: 0.10 },
            GaussianComponent { mean: 1200.0, std_dev: 150.0, weight: 0.05 },
        ],
        timing_dist: vec![
            (20.0, 0.70),
            (40.0, 0.15),
            (60.0, 0.10),
            (100.0, 0.05),
        ],
        port_range: (50000, 50100),
        protocol_signature: ProtocolSignature {
            header_pattern: vec![0x80, 0x78],
            typical_ports: vec![50000, 50001, 50002],
            encryption_type: EncryptionType::Dtls,
            direction_pattern: DirectionPattern::Symmetric,
        },
    };

    let spotify = ProtocolProfile {
        name: "spotify".to_string(),
        packet_size_dist: vec![
            GaussianComponent { mean: 512.0, std_dev: 128.0, weight: 0.45 },
            GaussianComponent { mean: 1024.0, std_dev: 256.0, weight: 0.30 },
            GaussianComponent { mean: 1400.0, std_dev: 100.0, weight: 0.20 },
            GaussianComponent { mean: 200.0, std_dev: 50.0, weight: 0.05 },
        ],
        timing_dist: vec![
            (10.0, 0.40),
            (25.0, 0.25),
            (50.0, 0.20),
            (100.0, 0.10),
            (200.0, 0.05),
        ],
        port_range: (4070, 4080),
        protocol_signature: ProtocolSignature {
            header_pattern: vec![0x00, 0x00],
            typical_ports: vec![4070, 4370, 4380],
            encryption_type: EncryptionType::Tls,
            direction_pattern: DirectionPattern::ServerHeavy,
        },
    };

    let generic_https = ProtocolProfile {
        name: "generic_https".to_string(),
        packet_size_dist: vec![
            GaussianComponent { mean: 256.0, std_dev: 80.0, weight: 0.30 },
            GaussianComponent { mean: 768.0, std_dev: 200.0, weight: 0.35 },
            GaussianComponent { mean: 1200.0, std_dev: 200.0, weight: 0.25 },
            GaussianComponent { mean: 1460.0, std_dev: 40.0, weight: 0.10 },
        ],
        timing_dist: vec![
            (5.0, 0.20),
            (15.0, 0.25),
            (30.0, 0.20),
            (75.0, 0.15),
            (150.0, 0.10),
            (300.0, 0.10),
        ],
        port_range: (443, 443),
        protocol_signature: ProtocolSignature {
            header_pattern: vec![0x16, 0x03, 0x03],
            typical_ports: vec![443, 8443],
            encryption_type: EncryptionType::Tls,
            direction_pattern: DirectionPattern::Bursty,
        },
    };

    let profiles = vec![cs2, discord_voice, spotify, generic_https];

    tracing::info!(
        profile_count = profiles.len(),
        profiles = ?profiles.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        "Default protocol profiles created"
    );

    Ok(profiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_profiles() -> Vec<ProtocolProfile> {
        vec![
            ProtocolProfile {
                name: "test_a".to_string(),
                packet_size_dist: vec![
                    GaussianComponent { mean: 100.0, std_dev: 20.0, weight: 0.6 },
                    GaussianComponent { mean: 300.0, std_dev: 50.0, weight: 0.4 },
                ],
                timing_dist: vec![
                    (20.0, 0.7),
                    (50.0, 0.3),
                ],
                port_range: (8000, 8100),
                protocol_signature: ProtocolSignature {
                    header_pattern: vec![0xAA, 0xBB],
                    typical_ports: vec![8000],
                    encryption_type: EncryptionType::Tls,
                    direction_pattern: DirectionPattern::ClientHeavy,
                },
            },
            ProtocolProfile {
                name: "test_b".to_string(),
                packet_size_dist: vec![
                    GaussianComponent { mean: 200.0, std_dev: 40.0, weight: 0.5 },
                    GaussianComponent { mean: 500.0, std_dev: 100.0, weight: 0.5 },
                ],
                timing_dist: vec![
                    (10.0, 0.5),
                    (30.0, 0.3),
                    (60.0, 0.2),
                ],
                port_range: (9000, 9100),
                protocol_signature: ProtocolSignature {
                    header_pattern: vec![0xCC, 0xDD],
                    typical_ports: vec![9000],
                    encryption_type: EncryptionType::Dtls,
                    direction_pattern: DirectionPattern::Symmetric,
                },
            },
        ]
    }

    #[test]
    fn test_blender_creation() {
        let profiles = test_profiles();
        let blender = ProtocolBlender::new(profiles).unwrap();
        assert_eq!(blender.profiles.len(), 2);
        assert_eq!(blender.blend_weights.len(), 2);
        assert!((blender.blend_weights[0] - 0.5).abs() < 0.01);
        assert!((blender.blend_weights[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blender_requires_min_profiles() {
        let profiles = vec![test_profiles()[0].clone()];
        let result = ProtocolBlender::new(profiles);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_protocol() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let idx = blender.select_protocol(&mut rng);
            assert!(idx < blender.profiles.len());
        }
    }

    #[test]
    fn test_adapt_weights_low_risk() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();

        blender.adapt_weights(0.1);

        let total: f64 = blender.blend_weights.iter().sum();
        assert!((total - 1.0).abs() < 0.01);

        for (i, &w) in blender.blend_weights.iter().enumerate() {
            assert!(w > 0.0, "Weight {} became non-positive", i);
        }
    }

    #[test]
    fn test_adapt_weights_high_risk() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();
        blender.current_protocol = 0;

        blender.adapt_weights(0.9);

        let total: f64 = blender.blend_weights.iter().sum();
        assert!((total - 1.0).abs() < 0.01);

        assert!(blender.blend_weights[0] < blender.blend_weights[1]);
    }

    #[test]
    fn test_blend_entropy() {
        let profiles = test_profiles();
        let blender = ProtocolBlender::new(profiles).unwrap();

        let entropy = blender.blend_entropy();
        let max_entropy = (2.0_f64).ln();

        assert!(entropy > 0.0);
        assert!(entropy <= max_entropy + 0.01);
    }

    #[test]
    fn test_blend_packet_padding() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        blender.current_protocol = 0;
        let small_data = vec![0x01, 0x02, 0x03];
        let result = blender.blend_packet(small_data, &mut rng).unwrap();

        assert!(result.data.len() >= 3);
        assert!(!result.protocol_hint.is_empty());
        assert!(result.delay.as_millis() >= 0);
    }

    #[test]
    fn test_blend_packet_truncation() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        blender.current_protocol = 0;
        let large_data = vec![0xAB; 2000];
        let result = blender.blend_packet(large_data, &mut rng).unwrap();

        assert!(result.data.len() <= 2000);
        assert!(!result.protocol_hint.is_empty());
    }

    #[test]
    fn test_protocol_statistics_update() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        blender.current_protocol = 0;
        for _ in 0..50 {
            let data = vec![0x00; 100];
            let _ = blender.blend_packet(data, &mut rng);
        }

        let stats = &blender.protocol_stats[0];
        assert!(stats.packet_count > 0);
        assert!(stats.mean_packet_size > 0.0);
    }

    #[test]
    fn test_default_profiles() {
        let profiles = create_default_profiles().unwrap();
        assert_eq!(profiles.len(), 4);

        let names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"cs2"));
        assert!(names.contains(&"discord_voice"));
        assert!(names.contains(&"spotify"));
        assert!(names.contains(&"generic_https"));
    }

    #[test]
    fn test_blend_chi_squared() {
        let profiles = test_profiles();
        let blender = ProtocolBlender::new(profiles).unwrap();

        let observed = vec![45, 55];
        let chi_sq = blender.compute_blend_chi_squared(&observed);
        assert!(chi_sq.is_finite());
        assert!(chi_sq >= 0.0);
    }

    #[test]
    fn test_reset_statistics() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..20 {
            let _ = blender.select_protocol(&mut rng);
        }

        blender.reset_statistics();
        assert_eq!(blender.switch_counter, 0);

        for stats in &blender.protocol_stats {
            assert_eq!(stats.packet_count, 0);
        }
    }

    #[test]
    fn test_profile_by_name() {
        let profiles = test_profiles();
        let blender = ProtocolBlender::new(profiles).unwrap();

        assert_eq!(blender.profile_by_name("test_a"), Some(0));
        assert_eq!(blender.profile_by_name("test_b"), Some(1));
        assert_eq!(blender.profile_by_name("nonexistent"), None);
    }

    #[test]
    fn test_direction_upload_probability() {
        let profiles = test_profiles();
        let blender = ProtocolBlender::new(profiles).unwrap();

        let client_heavy = blender.direction_upload_probability(&blender.profiles[0]);
        let symmetric = blender.direction_upload_probability(&blender.profiles[1]);

        assert!(client_heavy > 0.5);
        assert!((symmetric - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_weight_normalization_after_adapt() {
        let profiles = test_profiles();
        let mut blender = ProtocolBlender::new(profiles).unwrap();

        for _ in 0..100 {
            blender.adapt_weights(0.5);
            let total: f64 = blender.blend_weights.iter().sum();
            assert!((total - 1.0).abs() < 0.001, "Weights not normalized: {}", total);
        }
    }
}
