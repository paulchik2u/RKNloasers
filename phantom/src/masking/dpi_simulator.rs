use rand::Rng;
use rand_distr::{Distribution, Normal, Exponential};
use std::collections::VecDeque;
use std::time::Instant;

use crate::masking::adversarial::{PacketDirection, PacketRecord};
use crate::masking::statistical_model::StatisticalGameTrafficModel;
use crate::utils::{PhantomError, PhantomResult};

// ─── DPI Region Configurations ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpiRegion {
    Russia,
    China,
    Iran,
    Generic,
}

impl DpiRegion {
    fn detection_weights(&self) -> DpiWeights {
        match self {
            DpiRegion::Russia => DpiWeights {
                size_weight: 0.25,
                timing_weight: 0.20,
                entropy_weight: 0.15,
                direction_weight: 0.10,
                periodicity_weight: 0.10,
                signature_weight: 0.20,
            },
            DpiRegion::China => DpiWeights {
                size_weight: 0.20,
                timing_weight: 0.25,
                entropy_weight: 0.20,
                direction_weight: 0.10,
                periodicity_weight: 0.15,
                signature_weight: 0.10,
            },
            DpiRegion::Iran => DpiWeights {
                size_weight: 0.30,
                timing_weight: 0.15,
                entropy_weight: 0.10,
                direction_weight: 0.15,
                periodicity_weight: 0.05,
                signature_weight: 0.25,
            },
            DpiRegion::Generic => DpiWeights {
                size_weight: 0.25,
                timing_weight: 0.20,
                entropy_weight: 0.15,
                direction_weight: 0.10,
                periodicity_weight: 0.10,
                signature_weight: 0.20,
            },
        }
    }

    fn blocking_threshold(&self) -> f64 {
        match self {
            DpiRegion::Russia => 0.55,
            DpiRegion::China => 0.45,
            DpiRegion::Iran => 0.50,
            DpiRegion::Generic => 0.60,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DpiWeights {
    size_weight: f64,
    timing_weight: f64,
    entropy_weight: f64,
    direction_weight: f64,
    periodicity_weight: f64,
    signature_weight: f64,
}

// ─── Feature Extractors ───────────────────────────────────────────────────────

pub struct SizeFeatureExtractor {
    window: VecDeque<usize>,
    window_size: usize,
    ema_mean: f64,
    ema_variance: f64,
    alpha: f64,
}

impl SizeFeatureExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            ema_mean: 0.0,
            ema_variance: 0.0,
            alpha: 0.05,
        }
    }

    pub fn observe(&mut self, size: usize) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(size);

        let size_f64 = size as f64;
        if self.ema_mean == 0.0 {
            self.ema_mean = size_f64;
            self.ema_variance = 0.0;
        } else {
            let diff = size_f64 - self.ema_mean;
            self.ema_mean += self.alpha * diff;
            self.ema_variance =
                (1.0 - self.alpha) * (self.ema_variance + self.alpha * diff * diff);
        }
    }

    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().map(|&s| s as f64).sum::<f64>() / self.window.len() as f64
    }

    pub fn std_dev(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .window
            .iter()
            .map(|&s| {
                let diff = s as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.window.len() as f64;
        variance.sqrt()
    }

    pub fn skewness(&self) -> f64 {
        if self.window.len() < 3 {
            return 0.0;
        }
        let mean = self.mean();
        let std = self.std_dev();
        if std < 1e-6 {
            return 0.0;
        }
        let n = self.window.len() as f64;
        self.window
            .iter()
            .map(|&s| {
                let z = (s as f64 - mean) / std;
                z * z * z
            })
            .sum::<f64>()
            * n / ((n - 1.0) * (n - 2.0))
    }

    pub fn kurtosis(&self) -> f64 {
        if self.window.len() < 4 {
            return 3.0;
        }
        let mean = self.mean();
        let std = self.std_dev();
        if std < 1e-6 {
            return 3.0;
        }
        let n = self.window.len() as f64;
        let m4: f64 = self
            .window
            .iter()
            .map(|&s| {
                let z = (s as f64 - mean) / std;
                z * z * z * z
            })
            .sum::<f64>()
            / n;
        let m2: f64 = self
            .window
            .iter()
            .map(|&s| {
                let z = (s as f64 - mean) / std;
                z * z
            })
            .sum::<f64>()
            / n;
        if m2 < 1e-12 {
            return 3.0;
        }
        m4 / (m2 * m2)
    }

    pub fn percentile(&self, p: f64) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.window.iter().map(|&s| s as f64).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (p * (sorted.len() - 1) as f64) as usize;
        let idx_next = (idx + 1).min(sorted.len() - 1);
        let t = p * (sorted.len() - 1) as f64 - idx as f64;
        sorted[idx] * (1.0 - t) + sorted[idx_next] * t
    }

    pub fn coefficient_of_variation(&self) -> f64 {
        let mean = self.mean();
        if mean < 1e-6 {
            return 0.0;
        }
        self.std_dev() / mean
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.ema_mean = 0.0;
        self.ema_variance = 0.0;
    }
}

pub struct TimingFeatureExtractor {
    window: VecDeque<f64>,
    window_size: usize,
    ema_mean: f64,
    ema_variance: f64,
    alpha: f64,
}

impl TimingFeatureExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            ema_mean: 0.0,
            ema_variance: 0.0,
            alpha: 0.05,
        }
    }

    pub fn observe(&mut self, iat_ms: f64) {
        if iat_ms < 0.0 {
            return;
        }
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(iat_ms);

        if self.ema_mean == 0.0 {
            self.ema_mean = iat_ms;
            self.ema_variance = 0.0;
        } else {
            let diff = iat_ms - self.ema_mean;
            self.ema_mean += self.alpha * diff;
            self.ema_variance =
                (1.0 - self.alpha) * (self.ema_variance + self.alpha * diff * diff);
        }
    }

    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    pub fn std_dev(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .window
            .iter()
            .map(|&t| {
                let diff = t - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.window.len() as f64;
        variance.sqrt()
    }

    pub fn autocorrelation_lag1(&self) -> f64 {
        if self.window.len() < 3 {
            return 0.0;
        }
        let data: Vec<f64> = self.window.iter().copied().collect();
        let mean = self.mean();
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        if variance < 1e-12 {
            return 0.0;
        }
        let cov: f64 = (0..data.len() - 1)
            .map(|i| (data[i] - mean) * (data[i + 1] - mean))
            .sum::<f64>()
            / (data.len() - 1) as f64;
        (cov / variance).clamp(-1.0, 1.0)
    }

    pub fn burstiness_index(&self) -> f64 {
        let cv = self.coefficient_of_variation();
        let autocorr = self.autocorrelation_lag1().abs();
        (0.5 * cv + 0.5 * autocorr).clamp(0.0, 1.0)
    }

    pub fn coefficient_of_variation(&self) -> f64 {
        let mean = self.mean();
        if mean < 1e-6 {
            return 0.0;
        }
        self.std_dev() / mean
    }

    pub fn percentile(&self, p: f64) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (p * (sorted.len() - 1) as f64) as usize;
        let idx_next = (idx + 1).min(sorted.len() - 1);
        let t = p * (sorted.len() - 1) as f64 - idx as f64;
        sorted[idx] * (1.0 - t) + sorted[idx_next] * t
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.ema_mean = 0.0;
        self.ema_variance = 0.0;
    }
}

pub struct EntropyFeatureExtractor {
    window: VecDeque<f64>,
    window_size: usize,
    ema_mean: f64,
    ema_variance: f64,
    alpha: f64,
}

impl EntropyFeatureExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            ema_mean: 0.0,
            ema_variance: 0.0,
            alpha: 0.05,
        }
    }

    pub fn observe(&mut self, entropy: f64) {
        let clamped = entropy.clamp(0.0, 8.0);
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(clamped);

        if self.ema_mean == 0.0 {
            self.ema_mean = clamped;
            self.ema_variance = 0.0;
        } else {
            let diff = clamped - self.ema_mean;
            self.ema_mean += self.alpha * diff;
            self.ema_variance =
                (1.0 - self.alpha) * (self.ema_variance + self.alpha * diff * diff);
        }
    }

    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    pub fn std_dev(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self
            .window
            .iter()
            .map(|&e| {
                let diff = e - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.window.len() as f64;
        variance.sqrt()
    }

    pub fn is_high_entropy(&self) -> bool {
        self.mean() > 7.5
    }

    pub fn entropy_stability(&self) -> f64 {
        let std = self.std_dev();
        if self.mean() < 1e-6 {
            return 1.0;
        }
        (std / self.mean()).clamp(0.0, 1.0)
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.ema_mean = 0.0;
        self.ema_variance = 0.0;
    }
}

pub struct DirectionFeatureExtractor {
    bytes_up: u64,
    bytes_down: u64,
    packet_count_up: u64,
    packet_count_down: u64,
    ema_ratio: f64,
    alpha: f64,
}

impl DirectionFeatureExtractor {
    pub fn new() -> Self {
        Self {
            bytes_up: 0,
            bytes_down: 0,
            packet_count_up: 0,
            packet_count_down: 0,
            ema_ratio: 0.5,
            alpha: 0.05,
        }
    }

    pub fn observe(&mut self, size: usize, direction: PacketDirection) {
        match direction {
            PacketDirection::Up => {
                self.bytes_up += size as u64;
                self.packet_count_up += 1;
            }
            PacketDirection::Down => {
                self.bytes_down += size as u64;
                self.packet_count_down += 1;
            }
        }

        let total = self.bytes_up + self.bytes_down;
        if total > 0 {
            let current_ratio = self.bytes_up as f64 / total as f64;
            self.ema_ratio = self.alpha * current_ratio + (1.0 - self.alpha) * self.ema_ratio;
        }
    }

    pub fn byte_ratio(&self) -> f64 {
        let total = self.bytes_up + self.bytes_down;
        if total == 0 {
            return 0.5;
        }
        self.bytes_up as f64 / total as f64
    }

    pub fn packet_ratio(&self) -> f64 {
        let total = self.packet_count_up + self.packet_count_down;
        if total == 0 {
            return 0.5;
        }
        self.packet_count_up as f64 / total as f64
    }

    pub fn ema_ratio(&self) -> f64 {
        self.ema_ratio
    }

    pub fn total_bytes(&self) -> u64 {
        self.bytes_up + self.bytes_down
    }

    pub fn total_packets(&self) -> u64 {
        self.packet_count_up + self.packet_count_down
    }

    pub fn asymmetry_score(&self) -> f64 {
        let ratio = self.byte_ratio();
        (ratio - 0.5).abs() * 2.0
    }

    pub fn reset(&mut self) {
        self.bytes_up = 0;
        self.bytes_down = 0;
        self.packet_count_up = 0;
        self.packet_count_down = 0;
        self.ema_ratio = 0.5;
    }
}

pub struct PeriodicityExtractor {
    timestamps: VecDeque<f64>,
    window_size: usize,
}

impl PeriodicityExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn observe(&mut self, timestamp_ms: f64) {
        if self.timestamps.len() >= self.window_size {
            self.timestamps.pop_front();
        }
        self.timestamps.push_back(timestamp_ms);
    }

    pub fn periodicity_score(&self) -> f64 {
        if self.timestamps.len() < 8 {
            return 0.0;
        }

        let iats: Vec<f64> = self.timestamps.windows(2).filter_map(|w| {
            let dt = w[1] - w[0];
            if dt > 0.0 {
                Some(dt)
            } else {
                None
            }
        }).collect();

        if iats.len() < 4 {
            return 0.0;
        }

        let mean_iat = iats.iter().sum::<f64>() / iats.len() as f64;
        if mean_iat < 1e-6 {
            return 0.0;
        }

        let variance = iats
            .iter()
            .map(|&x| (x - mean_iat).powi(2))
            .sum::<f64>()
            / iats.len() as f64;
        let std_iat = variance.sqrt();
        let cv = std_iat / mean_iat;

        let autocorr = {
            let n = iats.len();
            let var = variance;
            if var < 1e-12 {
                0.0
            } else {
                let cov: f64 = (0..n - 1)
                    .map(|i| (iats[i] - mean_iat) * (iats[i + 1] - mean_iat))
                    .sum::<f64>()
                    / (n - 1) as f64;
                (cov / var).abs()
            }
        };

        let run_test_score = {
            let median = {
                let mut sorted = iats.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            };
            let above: Vec<bool> = iats.iter().map(|&x| x > median).collect();
            let mut runs = 1usize;
            for i in 1..above.len() {
                if above[i] != above[i - 1] {
                    runs += 1;
                }
            }
            let n_above = above.iter().filter(|&&x| x).count();
            let n_below = above.len() - n_above;
            if n_above == 0 || n_below == 0 {
                0.0
            } else {
                let expected =
                    (2.0 * n_above as f64 * n_below as f64) / above.len() as f64 + 1.0;
                let std_runs = ((2.0 * n_above as f64 * n_below as f64
                    * (2.0 * n_above as f64 * n_below as f64 - above.len() as f64))
                    / (above.len() as f64 * above.len() as f64 * (above.len() as f64 - 1.0)))
                .sqrt();
                if std_runs < 1e-6 {
                    0.0
                } else {
                    ((runs as f64 - expected) / std_runs).abs().min(4.0) / 4.0
                }
            }
        };

        (cv * 0.3 + (1.0 - cv).min(1.0) * autocorr * 0.4 + run_test_score * 0.3).clamp(0.0, 1.0)
    }

    pub fn dominant_period(&self) -> Option<f64> {
        if self.timestamps.len() < 16 {
            return None;
        }

        let iats: Vec<f64> = self.timestamps.windows(2).filter_map(|w| {
            let dt = w[1] - w[0];
            if dt > 0.0 {
                Some(dt)
            } else {
                None
            }
        }).collect();

        if iats.len() < 8 {
            return None;
        }

        let mean = iats.iter().sum::<f64>() / iats.len() as f64;
        Some(mean)
    }

    pub fn reset(&mut self) {
        self.timestamps.clear();
    }
}

// ─── Traffic Classifier ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrafficProfile {
    pub name: String,
    size_mean: f64,
    size_std: f64,
    timing_mean: f64,
    timing_std: f64,
    entropy_mean: f64,
    direction_ratio: f64,
    periodicity_score: f64,
}

impl TrafficProfile {
    pub fn new(
        name: &str,
        size_mean: f64,
        size_std: f64,
        timing_mean: f64,
        timing_std: f64,
        entropy_mean: f64,
        direction_ratio: f64,
        periodicity_score: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            size_mean,
            size_std: size_std.max(1e-6),
            timing_mean,
            timing_std: timing_std.max(1e-6),
            entropy_mean,
            direction_ratio,
            periodicity_score,
        }
    }

    fn mahalanobis_distance(
        &self,
        size: f64,
        timing: f64,
        entropy: f64,
        direction: f64,
        periodicity: f64,
    ) -> f64 {
        let size_z = ((size - self.size_mean) / self.size_std).powi(2);
        let timing_z = ((timing - self.timing_mean) / self.timing_std).powi(2);
        let entropy_z = ((entropy - self.entropy_mean) / 1.0).powi(2);
        let direction_z = (direction - self.direction_ratio).powi(2) * 4.0;
        let periodicity_z = (periodicity - self.periodicity_score).powi(2) * 4.0;

        let size_w = 0.30;
        let timing_w = 0.25;
        let entropy_w = 0.15;
        let direction_w = 0.15;
        let periodicity_w = 0.15;

        size_w * size_z + timing_w * timing_z + entropy_w * entropy_z
            + direction_w * direction_z + periodicity_w * periodicity_z
    }
}

pub struct TrafficClassifier {
    profiles: Vec<TrafficProfile>,
    threshold: f64,
}

impl TrafficClassifier {
    pub fn new(threshold: f64) -> Self {
        let profiles = Self::build_default_profiles();
        Self { profiles, threshold }
    }

    pub fn classify(
        &self,
        size_mean: f64,
        size_std: f64,
        timing_mean: f64,
        timing_std: f64,
        entropy_mean: f64,
        direction_ratio: f64,
        periodicity: f64,
    ) -> (String, f64) {
        if size_mean < 1e-6 {
            return ("unknown".to_string(), 0.0);
        }

        let mut best_name = "unknown".to_string();
        let mut best_score = f64::MAX;

        for profile in &self.profiles {
            let dist = profile.mahalanobis_distance(
                size_mean,
                size_std,
                timing_mean,
                timing_std,
                entropy_mean,
                direction_ratio,
                periodicity,
            );
            if dist < best_score {
                best_score = dist;
                best_name = profile.name.clone();
            }
        }

        let confidence = (-best_score / self.threshold).exp();
        (best_name, confidence.clamp(0.0, 1.0))
    }

    fn build_default_profiles() -> Vec<TrafficProfile> {
        vec![
            TrafficProfile::new(
                "openvpn", 800.0, 400.0, 50.0, 30.0, 7.8, 0.48, 0.1,
            ),
            TrafficProfile::new(
                "wireguard", 600.0, 350.0, 20.0, 15.0, 7.9, 0.50, 0.05,
            ),
            TrafficProfile::new(
                "shadowsocks", 500.0, 300.0, 40.0, 25.0, 7.9, 0.45, 0.08,
            ),
            TrafficProfile::new(
                "trojan", 700.0, 380.0, 60.0, 40.0, 7.6, 0.42, 0.12,
            ),
            TrafficProfile::new(
                "tor", 400.0, 250.0, 80.0, 60.0, 7.8, 0.50, 0.05,
            ),
            TrafficProfile::new(
                "cs2", 180.0, 120.0, 16.0, 10.0, 7.4, 0.33, 0.20,
            ),
            TrafficProfile::new(
                "valorant", 140.0, 90.0, 17.0, 12.0, 7.2, 0.35, 0.18,
            ),
            TrafficProfile::new(
                "minecraft", 400.0, 300.0, 50.0, 40.0, 6.8, 0.31, 0.15,
            ),
            TrafficProfile::new(
                "discord", 300.0, 200.0, 25.0, 20.0, 7.0, 0.40, 0.25,
            ),
            TrafficProfile::new(
                "https", 600.0, 500.0, 100.0, 80.0, 7.2, 0.35, 0.10,
            ),
            TrafficProfile::new(
                "youtube", 900.0, 600.0, 30.0, 20.0, 7.5, 0.15, 0.30,
            ),
            TrafficProfile::new(
                "zoom", 350.0, 200.0, 20.0, 15.0, 7.6, 0.50, 0.35,
            ),
        ]
    }

    pub fn add_profile(&mut self, profile: TrafficProfile) {
        self.profiles.push(profile);
    }
}

// ─── DPI Signatures ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TlsFingerprint {
    pub ja3_hash_prefix: Option<u32>,
    pub sni_patterns: Vec<String>,
    pub alpn_protocols: Vec<String>,
    pub cipher_suites: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct TimingPattern {
    pub mean_iat_ms: f64,
    pub std_iat_ms: f64,
    pub burst_interval_ms: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DpiSignature {
    pub name: String,
    pub port_pattern: Option<Vec<u16>>,
    pub tls_pattern: Option<TlsFingerprint>,
    pub timing_pattern: Option<TimingPattern>,
    pub severity: f64,
}

impl DpiSignature {
    pub fn matches_port(&self, port: u16) -> bool {
        self.port_pattern
            .as_ref()
            .map_or(false, |ports| ports.contains(&port))
    }

    pub fn matches_timing(&self, observed_iat: f64, observed_std: f64) -> f64 {
        match &self.timing_pattern {
            Some(pattern) => {
                let iat_diff = (observed_iat - pattern.mean_iat_ms).abs()
                    / pattern.mean_iat_ms.max(1e-6);
                let std_diff = if pattern.std_iat_ms > 0.0 {
                    (observed_std - pattern.std_iat_ms).abs() / pattern.std_iat_ms
                } else {
                    0.0
                };
                let match_score = 1.0 - (0.6 * iat_diff + 0.4 * std_diff).min(1.0);
                match_score.max(0.0)
            }
            None => 0.0,
        }
    }
}

// ─── Classification Result ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum TrafficType {
    Vpn,
    Tor,
    Proxy,
    Gaming,
    VoiceChat,
    Streaming,
    WebBrowsing,
    Unknown,
}

impl std::fmt::Display for TrafficType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrafficType::Vpn => write!(f, "VPN"),
            TrafficType::Tor => write!(f, "Tor"),
            TrafficType::Proxy => write!(f, "Proxy"),
            TrafficType::Gaming => write!(f, "Gaming"),
            TrafficType::VoiceChat => write!(f, "VoiceChat"),
            TrafficType::Streaming => write!(f, "Streaming"),
            TrafficType::WebBrowsing => write!(f, "WebBrowsing"),
            TrafficType::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Safe,
    LowRisk,
    MediumRisk,
    HighRisk,
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Safe => write!(f, "Safe"),
            RiskLevel::LowRisk => write!(f, "LowRisk"),
            RiskLevel::MediumRisk => write!(f, "MediumRisk"),
            RiskLevel::HighRisk => write!(f, "HighRisk"),
            RiskLevel::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub classification: TrafficType,
    pub confidence: f64,
    pub feature_vector: Vec<f64>,
    pub detected_as: String,
    pub risk_level: RiskLevel,
}

// ─── Analysis Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SizeAnalysis {
    pub observed_mean: f64,
    pub observed_std: f64,
    pub target_mean: f64,
    pub target_std: f64,
    pub z_score: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub detectability: f64,
}

#[derive(Debug, Clone)]
pub struct TimingAnalysis {
    pub observed_mean: f64,
    pub observed_std: f64,
    pub target_mean: f64,
    pub target_std: f64,
    pub burstiness: f64,
    pub autocorrelation: f64,
    pub detectability: f64,
}

#[derive(Debug, Clone)]
pub struct EntropyAnalysis {
    pub observed_mean: f64,
    pub observed_std: f64,
    pub target_mean: f64,
    pub stability: f64,
    pub is_suspicious: bool,
    pub detectability: f64,
}

#[derive(Debug, Clone)]
pub struct DirectionAnalysis {
    pub byte_ratio: f64,
    pub packet_ratio: f64,
    pub asymmetry: f64,
    pub target_ratio: f64,
    pub detectability: f64,
}

#[derive(Debug, Clone)]
pub struct PeriodicityAnalysis {
    pub score: f64,
    pub dominant_period_ms: Option<f64>,
    pub target_periodicity: f64,
    pub detectability: f64,
}

#[derive(Debug, Clone)]
pub struct DpiAnalysis {
    pub size_analysis: SizeAnalysis,
    pub timing_analysis: TimingAnalysis,
    pub entropy_analysis: EntropyAnalysis,
    pub direction_analysis: DirectionAnalysis,
    pub periodicity_analysis: PeriodicityAnalysis,
    pub overall_risk: RiskLevel,
    pub recommendations: Vec<String>,
}

// ─── DpiSimulator ─────────────────────────────────────────────────────────────

pub struct DpiSimulator {
    size_extractor: SizeFeatureExtractor,
    timing_extractor: TimingFeatureExtractor,
    entropy_extractor: EntropyFeatureExtractor,
    direction_extractor: DirectionFeatureExtractor,
    periodicity_extractor: PeriodicityExtractor,
    classifier: TrafficClassifier,
    dpi_signatures: Vec<DpiSignature>,
    region: DpiRegion,
    target_model: Option<StatisticalGameTrafficModel>,
    packet_count: u64,
    last_timestamp_ms: Option<f64>,
    flow_start: Option<Instant>,
}

impl DpiSimulator {
    pub fn new() -> Self {
        Self::with_region(DpiRegion::Generic)
    }

    pub fn with_region(region: DpiRegion) -> Self {
        tracing::info!(?region, "Initializing DpiSimulator");

        let signatures = Self::build_signature_database(region);

        Self {
            size_extractor: SizeFeatureExtractor::new(256),
            timing_extractor: TimingFeatureExtractor::new(256),
            entropy_extractor: EntropyFeatureExtractor::new(256),
            direction_extractor: DirectionFeatureExtractor::new(),
            periodicity_extractor: PeriodicityExtractor::new(256),
            classifier: TrafficClassifier::new(5.0),
            dpi_signatures: signatures,
            region,
            target_model: None,
            packet_count: 0,
            last_timestamp_ms: None,
            flow_start: None,
        }
    }

    pub fn with_target_model(mut self, model: StatisticalGameTrafficModel) -> Self {
        tracing::info!(model_name = %model.name, "Setting target traffic model for DPI simulation");
        self.target_model = Some(model);
        self
    }

    pub fn set_region(&mut self, region: DpiRegion) {
        tracing::info!(?region, "Switching DPI region profile");
        self.region = region;
        self.dpi_signatures = Self::build_signature_database(region);
    }

    pub fn process_packet(&mut self, packet: &PacketRecord) {
        let now_ms = packet
            .timestamp
            .duration_since(packet.timestamp)
            .as_secs_f64()
            * 1000.0;

        if self.flow_start.is_none() {
            self.flow_start = Some(Instant::now());
            self.last_timestamp_ms = Some(now_ms);
        }

        self.size_extractor.observe(packet.size);
        self.entropy_extractor.observe(packet.payload_entropy);
        self.direction_extractor.observe(packet.size, packet.direction);
        self.periodicity_extractor.observe(now_ms);

        if let Some(last_ts) = self.last_timestamp_ms {
            let iat = (now_ms - last_ts).abs();
            if iat > 0.0 {
                self.timing_extractor.observe(iat);
            }
        }
        self.last_timestamp_ms = Some(now_ms);

        self.packet_count += 1;

        if self.packet_count % 64 == 0 {
            let feature_vec = self.get_feature_vector();
            tracing::debug!(
                packet_count = self.packet_count,
                size_mean = feature_vec[0],
                timing_mean = feature_vec[1],
                entropy_mean = feature_vec[2],
                "DPI feature vector snapshot"
            );
        }
    }

    pub fn classify(&self) -> ClassificationResult {
        let feature_vec = self.get_feature_vector();
        let size_mean = feature_vec[0];
        let size_std = feature_vec[1];
        let timing_mean = feature_vec[2];
        let timing_std = feature_vec[3];
        let entropy_mean = feature_vec[4];
        let direction_ratio = feature_vec[5];
        let periodicity = feature_vec[6];

        let (detected_name, confidence) = self.classifier.classify(
            size_mean,
            size_std,
            timing_mean,
            timing_std,
            entropy_mean,
            direction_ratio,
            periodicity,
        );

        let signature_score = self.compute_signature_match();
        let combined_confidence = self.combine_confidence(confidence, signature_score);

        let classification = Self::map_to_traffic_type(&detected_name);
        let risk_level = self.compute_risk_level(combined_confidence, &classification);

        ClassificationResult {
            classification,
            confidence: combined_confidence,
            feature_vector: feature_vec,
            detected_as: detected_name,
            risk_level,
        }
    }

    pub fn get_feature_vector(&self) -> Vec<f64> {
        vec![
            self.size_extractor.mean(),
            self.size_extractor.std_dev(),
            self.timing_extractor.mean(),
            self.timing_extractor.std_dev(),
            self.entropy_extractor.mean(),
            self.direction_extractor.byte_ratio(),
            self.periodicity_extractor.periodicity_score(),
            self.size_extractor.skewness(),
            self.size_extractor.kurtosis(),
            self.timing_extractor.burstiness_index(),
            self.timing_extractor.autocorrelation_lag1(),
            self.entropy_extractor.entropy_stability(),
            self.direction_extractor.asymmetry_score(),
        ]
    }

    pub fn would_be_flagged(&self) -> bool {
        let result = self.classify();
        let threshold = self.region.blocking_threshold();
        result.confidence > threshold || result.risk_level >= RiskLevel::HighRisk
    }

    pub fn analyze(&self) -> DpiAnalysis {
        let feature_vec = self.get_feature_vector();
        let size_mean = feature_vec[0];
        let size_std = feature_vec[1];
        let timing_mean = feature_vec[2];
        let timing_std = feature_vec[3];
        let entropy_mean = feature_vec[4];
        let direction_ratio = feature_vec[5];
        let periodicity = feature_vec[6];

        let (target_size_mean, target_size_std, target_timing_mean, target_timing_std, target_entropy, target_direction, target_periodicity) =
            self.target_stats();

        let size_z = if target_size_std > 0.0 {
            (size_mean - target_size_mean) / target_size_std
        } else {
            0.0
        };
        let size_detectability = (size_z.abs() / 3.0).min(1.0);

        let timing_z = if target_timing_std > 0.0 {
            (timing_mean - target_timing_mean) / target_timing_std
        } else {
            0.0
        };
        let timing_detectability = (timing_z.abs() / 3.0).min(1.0);

        let entropy_detectability = if target_entropy > 0.0 {
            ((entropy_mean - target_entropy).abs() / target_entropy).min(1.0)
        } else {
            0.0
        };

        let direction_detectability = (direction_ratio - target_direction).abs() * 2.0;

        let periodicity_detectability = (periodicity - target_periodicity).abs().min(1.0);

        let overall_detectability = {
            let weights = self.region.detection_weights();
            weights.size_weight * size_detectability
                + weights.timing_weight * timing_detectability
                + weights.entropy_weight * entropy_detectability
                + weights.direction_weight * direction_detectability
                + weights.periodicity_weight * periodicity_detectability
                + weights.signature_weight * self.compute_signature_match()
        };

        let overall_risk = Self::detectability_to_risk(overall_detectability);
        let recommendations = self.generate_recommendations(
            size_detectability,
            timing_detectability,
            entropy_detectability,
            direction_detectability,
            periodicity_detectability,
        );

        DpiAnalysis {
            size_analysis: SizeAnalysis {
                observed_mean: size_mean,
                observed_std: size_std,
                target_mean: target_size_mean,
                target_std: target_size_std,
                z_score: size_z,
                skewness: self.size_extractor.skewness(),
                kurtosis: self.size_extractor.kurtosis(),
                detectability: size_detectability,
            },
            timing_analysis: TimingAnalysis {
                observed_mean: timing_mean,
                observed_std: timing_std,
                target_mean: target_timing_mean,
                target_std: target_timing_std,
                burstiness: self.timing_extractor.burstiness_index(),
                autocorrelation: self.timing_extractor.autocorrelation_lag1(),
                detectability: timing_detectability,
            },
            entropy_analysis: EntropyAnalysis {
                observed_mean: entropy_mean,
                observed_std: self.entropy_extractor.std_dev(),
                target_mean: target_entropy,
                stability: self.entropy_extractor.entropy_stability(),
                is_suspicious: self.entropy_extractor.is_high_entropy(),
                detectability: entropy_detectability,
            },
            direction_analysis: DirectionAnalysis {
                byte_ratio: direction_ratio,
                packet_ratio: self.direction_extractor.packet_ratio(),
                asymmetry: self.direction_extractor.asymmetry_score(),
                target_ratio: target_direction,
                detectability: direction_detectability,
            },
            periodicity_analysis: PeriodicityAnalysis {
                score: periodicity,
                dominant_period_ms: self.periodicity_extractor.dominant_period(),
                target_periodicity: target_periodicity,
                detectability: periodicity_detectability,
            },
            overall_risk,
            recommendations,
        }
    }

    pub fn reset(&mut self) {
        tracing::info!("Resetting DPI simulator state");
        self.size_extractor.reset();
        self.timing_extractor.reset();
        self.entropy_extractor.reset();
        self.direction_extractor.reset();
        self.periodicity_extractor.reset();
        self.packet_count = 0;
        self.last_timestamp_ms = None;
        self.flow_start = None;
    }

    pub fn packet_count(&self) -> u64 {
        self.packet_count
    }

    pub fn region(&self) -> DpiRegion {
        self.region
    }

    // ─── Private Helpers ──────────────────────────────────────────────────

    fn target_stats(&self) -> (f64, f64, f64, f64, f64, f64, f64) {
        match &self.target_model {
            Some(model) => (
                model.size_dist.mean,
                model.size_dist.std,
                model.timing_dist.mean_iat,
                model.timing_dist.std_iat,
                model.target_entropy,
                model.direction_ratio,
                model.periodicity_strength,
            ),
            None => (
                200.0,
                150.0,
                30.0,
                20.0,
                7.0,
                0.40,
                0.15,
            ),
        }
    }

    fn compute_signature_match(&self) -> f64 {
        if self.dpi_signatures.is_empty() || self.packet_count < 16 {
            return 0.0;
        }

        let timing_mean = self.timing_extractor.mean();
        let timing_std = self.timing_extractor.std_dev();

        let mut max_score = 0.0;
        for sig in &self.dpi_signatures {
            let timing_score = sig.matches_timing(timing_mean, timing_std);
            let combined = timing_score * sig.severity;
            if combined > max_score {
                max_score = combined;
            }
        }

        max_score.clamp(0.0, 1.0)
    }

    fn combine_confidence(&self, classifier_confidence: f64, signature_score: f64) -> f64 {
        let w_classifier = 0.65;
        let w_signature = 0.35;
        (w_classifier * classifier_confidence + w_signature * signature_score).clamp(0.0, 1.0)
    }

    fn compute_risk_level(&self, confidence: f64, traffic_type: &TrafficType) -> RiskLevel {
        let threshold = self.region.blocking_threshold();

        let is_blocked_protocol = matches!(
            traffic_type,
            TrafficType::Vpn | TrafficType::Tor | TrafficType::Proxy
        );

        if confidence > threshold + 0.20 && is_blocked_protocol {
            RiskLevel::Critical
        } else if confidence > threshold + 0.10 && is_blocked_protocol {
            RiskLevel::HighRisk
        } else if confidence > threshold {
            RiskLevel::MediumRisk
        } else if confidence > threshold - 0.15 {
            RiskLevel::LowRisk
        } else {
            RiskLevel::Safe
        }
    }

    fn detectability_to_risk(score: f64) -> RiskLevel {
        if score > 0.80 {
            RiskLevel::Critical
        } else if score > 0.60 {
            RiskLevel::HighRisk
        } else if score > 0.40 {
            RiskLevel::MediumRisk
        } else if score > 0.20 {
            RiskLevel::LowRisk
        } else {
            RiskLevel::Safe
        }
    }

    fn map_to_traffic_type(name: &str) -> TrafficType {
        match name {
            "openvpn" | "wireguard" => TrafficType::Vpn,
            "tor" => TrafficType::Tor,
            "shadowsocks" | "trojan" | "v2ray" => TrafficType::Proxy,
            "cs2" | "valorant" | "minecraft" | "fortnite" | "dota2" => TrafficType::Gaming,
            "discord" | "zoom" | "teams" | "skype" => TrafficType::VoiceChat,
            "youtube" | "netflix" | "twitch" => TrafficType::Streaming,
            "https" | "http" => TrafficType::WebBrowsing,
            _ => TrafficType::Unknown,
        }
    }

    fn generate_recommendations(
        &self,
        size_det: f64,
        timing_det: f64,
        entropy_det: f64,
        direction_det: f64,
        periodicity_det: f64,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if size_det > 0.5 {
            let (target_mean, target_std, _, _, _, _, _) = self.target_stats();
            recs.push(format!(
                "Packet size distribution diverges significantly (z={:.2}). \
                 Adjust padding to match target mean {:.0}±{:.0} bytes",
                (size_mean() - target_mean) / target_std.max(1e-6),
                target_mean,
                target_std
            ));
        }

        if timing_det > 0.5 {
            let (_, _, target_mean, target_std, _, _, _) = self.target_stats();
            let burstiness = self.timing_extractor.burstiness_index();
            recs.push(format!(
                "Timing pattern anomalous. Target IAT: {:.1}±{:.1}ms. \
                 Burstiness index: {:.2}. Apply gamma-distributed jitter",
                target_mean, target_std, burstiness
            ));
        }

        if entropy_det > 0.4 {
            let (_, _, _, _, target_entropy, _, _) = self.target_stats();
            let observed = self.entropy_extractor.mean();
            if observed > target_entropy + 0.3 {
                recs.push(format!(
                    "Entropy too high ({:.2} vs target {:.2}). \
                     Payload appears encrypted/random. Introduce structured padding",
                    observed, target_entropy
                ));
            } else if observed < target_entropy - 0.3 {
                recs.push(format!(
                    "Entropy too low ({:.2} vs target {:.2}). \
                     Payload may contain recognizable patterns",
                    observed, target_entropy
                ));
            }
        }

        if direction_det > 0.4 {
            let (_, _, _, _, _, target_dir, _) = self.target_stats();
            let observed = self.direction_extractor.byte_ratio();
            recs.push(format!(
                "Direction asymmetry detected (ratio {:.2} vs target {:.2}). \
                 Add dummy traffic in underrepresented direction",
                observed, target_dir
            ));
        }

        if periodicity_det > 0.5 {
            if let Some(period) = self.periodicity_extractor.dominant_period() {
                recs.push(format!(
                    "Strong periodicity detected (period {:.1}ms). \
                     DPI systems flag regular intervals. Apply phase-randomized jitter",
                    period
                ));
            } else {
                recs.push(
                    "Periodicity score elevated. Introduce non-deterministic \
                     inter-packet delays using exponential distribution"
                        .to_string(),
                );
            }
        }

        let sig_match = self.compute_signature_match();
        if sig_match > 0.3 {
            recs.push(format!(
                "Traffic matches known DPI signature (score {:.2}). \
                 Consider port randomization or TLS fingerprint obfuscation",
                sig_match
            ));
        }

        if recs.is_empty() {
            recs.push("Traffic profile within acceptable bounds for target model".to_string());
        }

        recs
    }

    fn build_signature_database(region: DpiRegion) -> Vec<DpiSignature> {
        let mut sigs = Vec::new();

        match region {
            DpiRegion::Russia => {
                sigs.push(DpiSignature {
                    name: "openvpn_russia".to_string(),
                    port_pattern: Some(vec![1194, 443, 8080, 8443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x1234),
                        sni_patterns: vec![],
                        alpn_protocols: vec!["openvpn".to_string()],
                        cipher_suites: vec![0x1301, 0x1302],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 50.0,
                        std_iat_ms: 30.0,
                        burst_interval_ms: Some(1000.0),
                    }),
                    severity: 0.9,
                });
                sigs.push(DpiSignature {
                    name: "wireguard_russia".to_string(),
                    port_pattern: Some(vec![51820, 51821, 443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 20.0,
                        std_iat_ms: 15.0,
                        burst_interval_ms: None,
                    }),
                    severity: 0.95,
                });
                sigs.push(DpiSignature {
                    name: "shadowsocks_russia".to_string(),
                    port_pattern: Some(vec![8388, 8389, 443, 8443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 40.0,
                        std_iat_ms: 25.0,
                        burst_interval_ms: Some(500.0),
                    }),
                    severity: 0.85,
                });
                sigs.push(DpiSignature {
                    name: "tor_russia".to_string(),
                    port_pattern: Some(vec![9001, 9030, 443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x5678),
                        sni_patterns: vec![],
                        alpn_protocols: vec![],
                        cipher_suites: vec![0x1301, 0x1302, 0x1303],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 80.0,
                        std_iat_ms: 60.0,
                        burst_interval_ms: Some(2000.0),
                    }),
                    severity: 0.95,
                });
                sigs.push(DpiSignature {
                    name: "trojan_russia".to_string(),
                    port_pattern: Some(vec![443, 8443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: None,
                        sni_patterns: vec![
                            "www.google.com".to_string(),
                            "cdn.cloudflare.com".to_string(),
                        ],
                        alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
                        cipher_suites: vec![0x1301, 0x1302],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 60.0,
                        std_iat_ms: 40.0,
                        burst_interval_ms: Some(800.0),
                    }),
                    severity: 0.80,
                });
            }
            DpiRegion::China => {
                sigs.push(DpiSignature {
                    name: "openvpn_gfw".to_string(),
                    port_pattern: Some(vec![1194, 443, 80]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x1234),
                        sni_patterns: vec![],
                        alpn_protocols: vec!["openvpn".to_string()],
                        cipher_suites: vec![0x1301],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 50.0,
                        std_iat_ms: 30.0,
                        burst_interval_ms: Some(1000.0),
                    }),
                    severity: 0.95,
                });
                sigs.push(DpiSignature {
                    name: "wireguard_gfw".to_string(),
                    port_pattern: Some(vec![51820, 443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 20.0,
                        std_iat_ms: 15.0,
                        burst_interval_ms: None,
                    }),
                    severity: 0.95,
                });
                sigs.push(DpiSignature {
                    name: "shadowsocks_gfw".to_string(),
                    port_pattern: Some(vec![8388, 8389, 443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 40.0,
                        std_iat_ms: 25.0,
                        burst_interval_ms: Some(500.0),
                    }),
                    severity: 0.90,
                });
                sigs.push(DpiSignature {
                    name: "v2ray_gfw".to_string(),
                    port_pattern: Some(vec![443, 8443, 8080]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: None,
                        sni_patterns: vec![],
                        alpn_protocols: vec!["h2".to_string()],
                        cipher_suites: vec![0x1301, 0x1302],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 35.0,
                        std_iat_ms: 20.0,
                        burst_interval_ms: Some(600.0),
                    }),
                    severity: 0.90,
                });
                sigs.push(DpiSignature {
                    name: "trojan_gfw".to_string(),
                    port_pattern: Some(vec![443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: None,
                        sni_patterns: vec![
                            "www.microsoft.com".to_string(),
                            "update.googleapis.com".to_string(),
                        ],
                        alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
                        cipher_suites: vec![0x1301, 0x1302],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 60.0,
                        std_iat_ms: 40.0,
                        burst_interval_ms: Some(800.0),
                    }),
                    severity: 0.85,
                });
                sigs.push(DpiSignature {
                    name: "tor_gfw".to_string(),
                    port_pattern: Some(vec![9001, 9030, 443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x5678),
                        sni_patterns: vec![],
                        alpn_protocols: vec![],
                        cipher_suites: vec![0x1301, 0x1302, 0x1303],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 80.0,
                        std_iat_ms: 60.0,
                        burst_interval_ms: Some(2000.0),
                    }),
                    severity: 0.95,
                });
            }
            DpiRegion::Iran => {
                sigs.push(DpiSignature {
                    name: "openvpn_iran".to_string(),
                    port_pattern: Some(vec![1194, 443, 8080]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x1234),
                        sni_patterns: vec![],
                        alpn_protocols: vec!["openvpn".to_string()],
                        cipher_suites: vec![0x1301],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 50.0,
                        std_iat_ms: 30.0,
                        burst_interval_ms: Some(1000.0),
                    }),
                    severity: 0.90,
                });
                sigs.push(DpiSignature {
                    name: "shadowsocks_iran".to_string(),
                    port_pattern: Some(vec![8388, 443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 40.0,
                        std_iat_ms: 25.0,
                        burst_interval_ms: Some(500.0),
                    }),
                    severity: 0.85,
                });
                sigs.push(DpiSignature {
                    name: "tor_iran".to_string(),
                    port_pattern: Some(vec![9001, 443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x5678),
                        sni_patterns: vec![],
                        alpn_protocols: vec![],
                        cipher_suites: vec![0x1301, 0x1302],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 80.0,
                        std_iat_ms: 60.0,
                        burst_interval_ms: Some(2000.0),
                    }),
                    severity: 0.95,
                });
            }
            DpiRegion::Generic => {
                sigs.push(DpiSignature {
                    name: "openvpn_generic".to_string(),
                    port_pattern: Some(vec![1194, 443]),
                    tls_pattern: Some(TlsFingerprint {
                        ja3_hash_prefix: Some(0x1234),
                        sni_patterns: vec![],
                        alpn_protocols: vec!["openvpn".to_string()],
                        cipher_suites: vec![0x1301],
                    }),
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 50.0,
                        std_iat_ms: 30.0,
                        burst_interval_ms: Some(1000.0),
                    }),
                    severity: 0.80,
                });
                sigs.push(DpiSignature {
                    name: "wireguard_generic".to_string(),
                    port_pattern: Some(vec![51820, 443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 20.0,
                        std_iat_ms: 15.0,
                        burst_interval_ms: None,
                    }),
                    severity: 0.80,
                });
                sigs.push(DpiSignature {
                    name: "shadowsocks_generic".to_string(),
                    port_pattern: Some(vec![8388, 443]),
                    tls_pattern: None,
                    timing_pattern: Some(TimingPattern {
                        mean_iat_ms: 40.0,
                        std_iat_ms: 25.0,
                        burst_interval_ms: Some(500.0),
                    }),
                    severity: 0.70,
                });
            }
        }

        tracing::info!(
            region = ?region,
            signature_count = sigs.len(),
            "Built DPI signature database"
        );

        sigs
    }
}

impl Default for DpiSimulator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helper function for recommendations ──────────────────────────────────────

fn size_mean() -> f64 {
    0.0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_test_packet(size: usize, direction: PacketDirection, entropy: f64) -> PacketRecord {
        PacketRecord {
            size,
            direction,
            timestamp: Instant::now(),
            payload_entropy: entropy,
        }
    }

    fn feed_gaming_traffic(sim: &mut DpiSimulator, count: usize) {
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(150.0, 50.0).unwrap();
        let timing_normal = Normal::new(16.0, 8.0).unwrap();
        let entropy_normal = Normal::new(7.2, 0.3).unwrap();

        for i in 0..count {
            let size = normal.sample(&mut rng).clamp(60.0, 300.0) as usize;
            let entropy = entropy_normal.sample(&mut rng).clamp(0.0, 8.0);
            let direction = if i % 3 == 0 {
                PacketDirection::Up
            } else {
                PacketDirection::Down
            };
            let packet = make_test_packet(size, direction, entropy);
            sim.process_packet(&packet);

            let iat = timing_normal.sample(&mut rng).clamp(1.0, 100.0);
            sim.timing_extractor.observe(iat);
        }
    }

    #[test]
    fn test_simulator_creation() {
        let sim = DpiSimulator::new();
        assert_eq!(sim.packet_count(), 0);
        assert_eq!(sim.region(), DpiRegion::Generic);
    }

    #[test]
    fn test_simulator_with_region() {
        let sim = DpiSimulator::with_region(DpiRegion::Russia);
        assert_eq!(sim.region(), DpiRegion::Russia);
        assert!(!sim.dpi_signatures.is_empty());
    }

    #[test]
    fn test_process_packet_updates_state() {
        let mut sim = DpiSimulator::new();
        let packet = make_test_packet(150, PacketDirection::Up, 7.2);
        sim.process_packet(&packet);
        assert_eq!(sim.packet_count(), 1);
    }

    #[test]
    fn test_feature_vector_has_correct_length() {
        let mut sim = DpiSimulator::new();
        feed_gaming_traffic(&mut sim, 64);
        let fv = sim.get_feature_vector();
        assert_eq!(fv.len(), 13);
    }

    #[test]
    fn test_classification_returns_valid_result() {
        let mut sim = DpiSimulator::new();
        feed_gaming_traffic(&mut sim, 128);
        let result = sim.classify();
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(!result.detected_as.is_empty());
    }

    #[test]
    fn test_would_be_flagged_for_vpn_traffic() {
        let mut sim = DpiSimulator::with_region(DpiRegion::China);

        let vpn_normal = Normal::new(700.0, 350.0).unwrap();
        let vpn_timing = Normal::new(50.0, 30.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..128 {
            let size = vpn_normal.sample(&mut rng).clamp(100.0, 1400.0) as usize;
            let direction = if i % 2 == 0 {
                PacketDirection::Up
            } else {
                PacketDirection::Down
            };
            let packet = make_test_packet(size, direction, 7.9);
            sim.process_packet(&packet);
            let iat = vpn_timing.sample(&mut rng).clamp(1.0, 200.0);
            sim.timing_extractor.observe(iat);
        }

        let _flagged = sim.would_be_flagged();
    }

    #[test]
    fn test_analysis_returns_valid_structure() {
        let mut sim = DpiSimulator::new();
        feed_gaming_traffic(&mut sim, 128);
        let analysis = sim.analyze();
        assert!(analysis.size_analysis.observed_mean > 0.0);
        assert!(analysis.timing_analysis.observed_mean > 0.0);
        assert!(analysis.entropy_analysis.observed_mean > 0.0);
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sim = DpiSimulator::new();
        feed_gaming_traffic(&mut sim, 64);
        assert!(sim.packet_count() > 0);
        sim.reset();
        assert_eq!(sim.packet_count(), 0);
        assert_eq!(sim.get_feature_vector()[0], 0.0);
    }

    #[test]
    fn test_size_extractor_statistics() {
        let mut ext = SizeFeatureExtractor::new(128);
        for _ in 0..100 {
            ext.observe(150);
        }
        assert!((ext.mean() - 150.0).abs() < 1.0);
        assert!(ext.std_dev() < 1.0);
    }

    #[test]
    fn test_timing_extractor_burstiness() {
        let mut ext = TimingFeatureExtractor::new(128);
        for i in 0..50 {
            let iat = if i % 5 == 0 { 100.0 } else { 10.0 };
            ext.observe(iat);
        }
        assert!(ext.burstiness_index() > 0.0);
    }

    #[test]
    fn test_entropy_extractor_high_entropy() {
        let mut ext = EntropyFeatureExtractor::new(128);
        for _ in 0..50 {
            ext.observe(7.9);
        }
        assert!(ext.is_high_entropy());
    }

    #[test]
    fn test_direction_extractor_asymmetry() {
        let mut ext = DirectionFeatureExtractor::new();
        for _ in 0..50 {
            ext.observe(200, PacketDirection::Down);
        }
        for _ in 0..10 {
            ext.observe(100, PacketDirection::Up);
        }
        assert!(ext.asymmetry_score() > 0.3);
    }

    #[test]
    fn test_periodicity_extractor_detects_regular_pattern() {
        let mut ext = PeriodicityExtractor::new(128);
        for i in 0..50 {
            ext.observe(i as f64 * 16.67);
        }
        assert!(ext.periodicity_score() > 0.0);
    }

    #[test]
    fn test_classifier_identifies_gaming_profile() {
        let classifier = TrafficClassifier::new(5.0);
        let (name, confidence) = classifier.classify(
            150.0, 80.0, 16.0, 10.0, 7.2, 0.35, 0.18,
        );
        assert!(name == "valorant" || name == "cs2");
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_classifier_identifies_vpn_profile() {
        let classifier = TrafficClassifier::new(5.0);
        let (name, confidence) = classifier.classify(
            700.0, 350.0, 50.0, 30.0, 7.8, 0.48, 0.1,
        );
        assert!(name == "openvpn" || name == "wireguard");
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Safe < RiskLevel::LowRisk);
        assert!(RiskLevel::LowRisk < RiskLevel::MediumRisk);
        assert!(RiskLevel::MediumRisk < RiskLevel::HighRisk);
        assert!(RiskLevel::HighRisk < RiskLevel::Critical);
    }

    #[test]
    fn test_dpi_signatures_match_timing() {
        let sig = DpiSignature {
            name: "test".to_string(),
            port_pattern: Some(vec![443]),
            tls_pattern: None,
            timing_pattern: Some(TimingPattern {
                mean_iat_ms: 50.0,
                std_iat_ms: 30.0,
                burst_interval_ms: Some(1000.0),
            }),
            severity: 0.9,
        };

        let close_match = sig.matches_timing(52.0, 28.0);
        let far_match = sig.matches_timing(200.0, 100.0);
        assert!(close_match > far_match);
    }

    #[test]
    fn test_dpi_signatures_match_port() {
        let sig = DpiSignature {
            name: "test".to_string(),
            port_pattern: Some(vec![1194, 443]),
            tls_pattern: None,
            timing_pattern: None,
            severity: 0.8,
        };
        assert!(sig.matches_port(1194));
        assert!(sig.matches_port(443));
        assert!(!sig.matches_port(8080));
    }

    #[test]
    fn test_region_blocking_thresholds() {
        assert!(DpiRegion::China.blocking_threshold() < DpiRegion::Russia.blocking_threshold());
        assert!(DpiRegion::Iran.blocking_threshold() < DpiRegion::Generic.blocking_threshold());
    }

    #[test]
    fn test_region_detection_weights_sum_to_one() {
        for region in [
            DpiRegion::Russia,
            DpiRegion::China,
            DpiRegion::Iran,
            DpiRegion::Generic,
        ] {
            let w = region.detection_weights();
            let total = w.size_weight
                + w.timing_weight
                + w.entropy_weight
                + w.direction_weight
                + w.periodicity_weight
                + w.signature_weight;
            assert!((total - 1.0).abs() < 0.01, "Weights for {:?} sum to {}", region, total);
        }
    }

    #[test]
    fn test_simulator_with_target_model() {
        let profile = crate::masking::profiles::GamingProfile {
            name: "test".to_string(),
            packet_size_min: 60,
            packet_size_max: 300,
            heartbeat_interval_ms: 30000,
            jitter_ms: 50,
            frequency_hz: 60,
        };
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let sim = DpiSimulator::new().with_target_model(model);
        feed_gaming_traffic(&mut DpiSimulator::try_clone_for_test(&sim), 64);
    }

    #[test]
    fn test_traffic_type_display() {
        assert_eq!(format!("{}", TrafficType::Vpn), "VPN");
        assert_eq!(format!("{}", TrafficType::Tor), "Tor");
        assert_eq!(format!("{}", TrafficType::Gaming), "Gaming");
    }

    #[test]
    fn test_risk_level_display() {
        assert_eq!(format!("{}", RiskLevel::Safe), "Safe");
        assert_eq!(format!("{}", RiskLevel::Critical), "Critical");
    }

    #[test]
    fn test_entropy_stability() {
        let mut ext = EntropyFeatureExtractor::new(128);
        for _ in 0..50 {
            ext.observe(7.0);
        }
        assert!(ext.entropy_stability() < 0.1);
    }

    #[test]
    fn test_size_percentile_ordering() {
        let mut ext = SizeFeatureExtractor::new(128);
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(200.0, 80.0).unwrap();
        for _ in 0..100 {
            ext.observe(normal.sample(&mut rng).clamp(40.0, 1500.0) as usize);
        }
        let p25 = ext.percentile(0.25);
        let p50 = ext.percentile(0.50);
        let p75 = ext.percentile(0.75);
        assert!(p25 <= p50);
        assert!(p50 <= p75);
    }

    #[test]
    fn test_timing_percentile_ordering() {
        let mut ext = TimingFeatureExtractor::new(128);
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(30.0, 15.0).unwrap();
        for _ in 0..100 {
            ext.observe(normal.sample(&mut rng).clamp(1.0, 200.0));
        }
        let p50 = ext.percentile(0.50);
        let p95 = ext.percentile(0.95);
        assert!(p50 <= p95);
    }

    #[test]
    fn test_direction_total_counts() {
        let mut ext = DirectionFeatureExtractor::new();
        for _ in 0..30 {
            ext.observe(100, PacketDirection::Up);
        }
        for _ in 0..20 {
            ext.observe(200, PacketDirection::Down);
        }
        assert_eq!(ext.total_packets(), 50);
        assert_eq!(ext.total_bytes(), 7000);
    }

    #[test]
    fn test_periodicity_dominant_period() {
        let mut ext = PeriodicityExtractor::new(128);
        for i in 0..30 {
            ext.observe(i as f64 * 25.0);
        }
        let period = ext.dominant_period();
        assert!(period.is_some());
        let p = period.unwrap();
        assert!((p - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_empty_sim_classification() {
        let sim = DpiSimulator::new();
        let result = sim.classify();
        assert_eq!(result.detected_as, "unknown");
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_empty_sim_analysis() {
        let sim = DpiSimulator::new();
        let analysis = sim.analyze();
        assert_eq!(analysis.size_analysis.observed_mean, 0.0);
        assert_eq!(analysis.timing_analysis.observed_mean, 0.0);
    }

    #[test]
    fn test_set_region_updates_signatures() {
        let mut sim = DpiSimulator::new();
        let initial_count = sim.dpi_signatures.len();
        sim.set_region(DpiRegion::China);
        assert_ne!(sim.dpi_signatures.len(), initial_count);
        assert_eq!(sim.region(), DpiRegion::China);
    }

    #[test]
    fn test_feature_vector_all_finite() {
        let mut sim = DpiSimulator::new();
        feed_gaming_traffic(&mut sim, 64);
        let fv = sim.get_feature_vector();
        for (i, &val) in fv.iter().enumerate() {
            assert!(val.is_finite(), "Feature {} is not finite: {}", i, val);
        }
    }

    #[test]
    fn test_recommendations_not_empty_after_traffic() {
        let mut sim = DpiSimulator::new();
        feed_gaming_traffic(&mut sim, 128);
        let analysis = sim.analyze();
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_detectability_to_risk_mapping() {
        assert_eq!(DpiSimulator::detectability_to_risk(0.0), RiskLevel::Safe);
        assert_eq!(DpiSimulator::detectability_to_risk(0.1), RiskLevel::Safe);
        assert_eq!(DpiSimulator::detectability_to_risk(0.3), RiskLevel::LowRisk);
        assert_eq!(DpiSimulator::detectability_to_risk(0.5), RiskLevel::MediumRisk);
        assert_eq!(DpiSimulator::detectability_to_risk(0.7), RiskLevel::HighRisk);
        assert_eq!(DpiSimulator::detectability_to_risk(0.9), RiskLevel::Critical);
    }
}
