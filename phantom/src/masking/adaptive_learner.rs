use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use crate::masking::adversarial::PacketRecord;
use crate::utils::{PhantomError, PhantomResult};

const DEFAULT_BIN_COUNT: usize = 64;
const DEFAULT_ANOMALY_THRESHOLD: f64 = 3.0;
const DEFAULT_ANOMALY_WINDOW: usize = 64;
const MIN_BURST_CLUSTER_SIZE: usize = 5;
const BURST_GAP_MS: f64 = 50.0;
const MAX_BURST_PATTERNS: usize = 16;
const PROFILE_CONFIDENCE_DECAY: f64 = 0.999;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub name: String,
    pub size_distribution: EmpiricalDistribution,
    pub timing_distribution: EmpiricalDistribution,
    pub burst_patterns: Vec<LearnedBurstPattern>,
    pub session_duration_dist: EmpiricalDistribution,
    pub idle_period_dist: EmpiricalDistribution,
    pub up_down_ratio: f64,
    pub payload_entropy: f64,
    pub sample_count: usize,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalDistribution {
    pub bins: Vec<f64>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedBurstPattern {
    pub name: String,
    pub avg_packet_count: f64,
    pub avg_inter_packet_time: f64,
    pub avg_packet_size: f64,
    pub frequency: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct OnlineStatisticsUpdater {
    count: usize,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    ema_mean: f64,
    ema_variance: f64,
    alpha: f64,
}

#[derive(Debug, Clone)]
pub struct ProfileSimilarityChecker {
    ks_significance: f64,
    chi_sq_significance: f64,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    threshold: f64,
    recent_scores: VecDeque<f64>,
    mean_score: f64,
    score_m2: f64,
    score_count: usize,
}

pub struct AdaptiveTrafficLearner {
    observed_profiles: HashMap<String, UserProfile>,
    active_profile: Option<String>,
    learning_rate: f64,
    min_samples: usize,
    online_updater: OnlineStatisticsUpdater,
    similarity_checker: ProfileSimilarityChecker,
    anomaly_detector: AnomalyDetector,
    raw_size_samples: HashMap<String, Vec<f64>>,
    raw_timing_samples: HashMap<String, Vec<f64>>,
    raw_burst_data: HashMap<String, Vec<BurstObservation>>,
    raw_session_durations: HashMap<String, Vec<f64>>,
    raw_idle_periods: HashMap<String, Vec<f64>>,
    raw_entropy_samples: HashMap<String, Vec<f64>>,
    direction_counts: HashMap<String, (usize, usize)>,
    current_burst: Option<BurstObservation>,
    last_packet_time: Option<std::time::Instant>,
    session_start: Option<std::time::Instant>,
    total_observations: usize,
    anomaly_count: usize,
}

#[derive(Debug, Clone)]
struct BurstObservation {
    packet_count: usize,
    inter_packet_times: Vec<f64>,
    packet_sizes: Vec<f64>,
}

pub struct LearningStats {
    pub total_observations: usize,
    pub profiles_learned: usize,
    pub average_confidence: f64,
    pub anomaly_rate: f64,
    pub profile_details: Vec<ProfileDetail>,
}

pub struct ProfileDetail {
    pub name: String,
    pub sample_count: usize,
    pub confidence: f64,
    pub size_mean: f64,
    pub size_std: f64,
    pub timing_mean: f64,
    pub timing_std: f64,
}

pub struct MaskedPacket {
    pub data: Vec<u8>,
    pub delay: Duration,
    pub context: String,
}

impl OnlineStatisticsUpdater {
    pub fn new(alpha: f64) -> Self {
        let clamped_alpha = alpha.clamp(0.001, 0.5);
        OnlineStatisticsUpdater {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            ema_mean: 0.0,
            ema_variance: 0.0,
            alpha: clamped_alpha,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let n = self.count as f64;
        let delta = value - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.mean += delta_n;

        if self.count == 1 {
            self.ema_mean = value;
            self.ema_variance = 0.0;
        } else {
            let a = self.alpha;
            let ema_delta = value - self.ema_mean;
            self.ema_mean += a * ema_delta;
            self.ema_variance =
                (1.0 - a) * (self.ema_variance + a * ema_delta * ema_delta);
        }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count as f64 - 1.0)
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn skewness(&self) -> f64 {
        let var = self.variance();
        if var < f64::MIN_POSITIVE || self.count < 3 {
            return 0.0;
        }
        let std = var.sqrt();
        let n = self.count as f64;
        (n / ((n - 1.0) * (n - 2.0))).sqrt() * (self.m3 / (std.powi(3) * n))
    }

    pub fn kurtosis(&self) -> f64 {
        let var = self.variance();
        if var < f64::MIN_POSITIVE || self.count < 4 {
            return 0.0;
        }
        let std = var.sqrt();
        let n = self.count as f64;
        let m4_term = self.m4 / (std.powi(4) * n);
        let correction = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
        m4_term * n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) - correction
    }

    pub fn ema_mean(&self) -> f64 {
        self.ema_mean
    }

    pub fn ema_std_dev(&self) -> f64 {
        self.ema_variance.sqrt()
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }
}

impl ProfileSimilarityChecker {
    pub fn new() -> Self {
        ProfileSimilarityChecker {
            ks_significance: 0.05,
            chi_sq_significance: 0.05,
        }
    }

    pub fn kolmogorov_smirnov(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 1.0;
        }
        let mut sorted_a = a.to_vec();
        let mut sorted_b = b.to_vec();
        sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

        let mut max_diff = 0.0f64;
        let n = sorted_a.len();
        let m = sorted_b.len();
        let all_values: Vec<f64> = sorted_a.iter().chain(sorted_b.iter()).copied().collect();

        let mut i = 0usize;
        let mut j = 0usize;
        let mut cum_a = 0usize;
        let mut cum_b = 0usize;

        for &val in &all_values {
            while i < n && sorted_a[i] <= val {
                cum_a += 1;
                i += 1;
            }
            while j < m && sorted_b[j] <= val {
                cum_b += 1;
                j += 1;
            }
            let fa = cum_a as f64 / n as f64;
            let fb = cum_b as f64 / m as f64;
            let diff = (fa - fb).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        max_diff
    }

    pub fn chi_squared_test(&self, observed: &[usize], expected: &[f64]) -> f64 {
        if observed.len() != expected.len() || observed.is_empty() {
            return f64::INFINITY;
        }
        let mut chi_sq = 0.0f64;
        for (o, e) in observed.iter().zip(expected.iter()) {
            if *e > 0.0 {
                let diff = *o as f64 - e;
                chi_sq += (diff * diff) / e;
            }
        }
        chi_sq
    }

    pub fn wasserstein_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return f64::INFINITY;
        }
        let mut sorted_a = a.to_vec();
        let mut sorted_b = b.to_vec();
        sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

        let max_len = sorted_a.len().max(sorted_b.len());
        let mut distance = 0.0f64;

        for i in 0..max_len {
            let va = if i < sorted_a.len() {
                sorted_a[i]
            } else {
                *sorted_a.last().unwrap_or(&0.0)
            };
            let vb = if i < sorted_b.len() {
                sorted_b[i]
            } else {
                *sorted_b.last().unwrap_or(&0.0)
            };
            distance += (va - vb).abs();
        }

        distance / max_len as f64
    }

    pub fn distribution_similarity(&self, a: &EmpiricalDistribution, b: &EmpiricalDistribution) -> f64 {
        let mean_diff = (a.mean - b.mean).abs() / (a.mean.abs().max(b.mean.abs()).max(1e-6));
        let std_diff = {
            let std_a = a.variance.sqrt();
            let std_b = b.variance.sqrt();
            (std_a - std_b).abs() / (std_a.max(std_b).max(1e-6))
        };
        let percentile_diff = (a.p50 - b.p50).abs() / (a.p50.abs().max(b.p50.abs()).max(1e-6));

        let combined = (mean_diff * 0.3 + std_diff * 0.3 + percentile_diff * 0.4).clamp(0.0, 10.0);
        1.0 - (combined / 10.0)
    }
}

impl Default for ProfileSimilarityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector {
    pub fn new(threshold: f64, window_size: usize) -> Self {
        AnomalyDetector {
            threshold: threshold.max(1.0),
            recent_scores: VecDeque::with_capacity(window_size.max(16)),
            mean_score: 0.0,
            score_m2: 0.0,
            score_count: 0,
        }
    }

    pub fn add_score(&mut self, score: f64) {
        if self.recent_scores.len() >= self.recent_scores.capacity() {
            let old = self.recent_scores.pop_front().unwrap();
            if self.score_count > 1 {
                let old_mean = self.mean_score;
                self.mean_score = (self.mean_score * self.score_count as f64 - old)
                    / (self.score_count as f64 - 1.0);
                let delta = old - old_mean;
                let delta_new = old - self.mean_score;
                self.score_m2 -= delta * delta_new;
                self.score_count -= 1;
            }
        }

        self.recent_scores.push_back(score);
        self.score_count += 1;
        let n = self.score_count as f64;
        let delta = score - self.mean_score;
        self.mean_score += delta / n;
        let delta2 = score - self.mean_score;
        self.score_m2 += delta * delta2;
    }

    pub fn is_anomalous(&self) -> bool {
        if self.recent_scores.len() < 8 {
            return false;
        }
        let variance = if self.score_count > 1 {
            self.score_m2 / (self.score_count as f64 - 1.0)
        } else {
            0.0
        };
        let std = variance.sqrt();
        if std < 1e-6 {
            return false;
        }
        if let Some(&latest) = self.recent_scores.back() {
            let z_score = (latest - self.mean_score).abs() / std;
            z_score > self.threshold
        } else {
            false
        }
    }

    pub fn anomaly_score(&self) -> f64 {
        if self.recent_scores.is_empty() {
            return 0.0;
        }
        let variance = if self.score_count > 1 {
            self.score_m2 / (self.score_count as f64 - 1.0)
        } else {
            0.0
        };
        let std = variance.sqrt();
        if std < 1e-6 {
            return 0.0;
        }
        if let Some(&latest) = self.recent_scores.back() {
            (latest - self.mean_score).abs() / std
        } else {
            0.0
        }
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold.max(1.0);
    }
}

impl EmpiricalDistribution {
    pub fn new(bin_count: usize) -> Self {
        let n = bin_count.max(8);
        EmpiricalDistribution {
            bins: vec![0.0; n],
            bin_edges: Vec::with_capacity(n + 1),
            total_count: 0,
            mean: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            p10: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }

    pub fn add_sample(&mut self, value: f64) {
        self.total_count += 1;
        let n = self.total_count as f64;
        let delta = value - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        if self.total_count >= 4 {
            self.kurtosis += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
                + 6.0 * delta_n2 * self.variance
                - 4.0 * delta_n * self.skewness;
        }
        if self.total_count >= 3 {
            self.skewness += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.variance;
        }
        self.variance += term1;
        self.mean += delta_n;

        self.update_histogram(value);
        self.update_percentiles_from_moments();
    }

    pub fn merge_samples(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }
        for &s in samples {
            self.add_sample(s);
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        if self.total_count == 0 || self.bins.is_empty() {
            return self.mean;
        }

        let total_weight: f64 = self.bins.iter().sum();
        if total_weight < f64::MIN_POSITIVE {
            return self.mean;
        }

        let mut roll: f64 = rng.gen_range(0.0..total_weight);
        let mut cumulative = 0.0;

        for (i, &weight) in self.bins.iter().enumerate() {
            cumulative += weight;
            if roll <= cumulative {
                let bin_idx = i;
                let bin_count = self.bins.len();

                if bin_idx == 0 {
                    let low = self.bin_edges.first().copied().unwrap_or(0.0);
                    let high = self.bin_edges.get(1).copied().unwrap_or(low + 1.0);
                    return self.interpolate_in_bin(low, high, rng);
                } else if bin_idx >= bin_count - 1 {
                    let low = self.bin_edges.get(bin_idx).copied().unwrap_or(0.0);
                    let high = self.bin_edges.last().copied().unwrap_or(low + 1.0);
                    return self.interpolate_in_bin(low, high, rng);
                } else {
                    let low = self.bin_edges.get(bin_idx).copied().unwrap_or(0.0);
                    let high = self.bin_edges.get(bin_idx + 1).copied().unwrap_or(low + 1.0);
                    return self.interpolate_in_bin(low, high, rng);
                }
            }
        }

        let low = self.bin_edges.last().copied().unwrap_or(0.0);
        let high = if self.bin_edges.len() >= 2 {
            *self.bin_edges.last().unwrap()
                + (self.bin_edges.last().unwrap() - self.bin_edges.iter().rev().nth(1).unwrap_or(&0.0))
        } else {
            low + 1.0
        };
        self.interpolate_in_bin(low, high, rng)
    }

    fn interpolate_in_bin(&self, low: f64, high: f64, rng: &mut impl Rng) -> f64 {
        let range = high - low;
        if range < 1e-6 {
            return low;
        }
        let u: f64 = rng.gen();
        low + u * range
    }

    fn update_histogram(&mut self, value: f64) {
        if self.bin_edges.is_empty() {
            self.bin_edges.push(value);
            self.bin_edges.push(value + 1.0);
            self.bins = vec![0.0; self.bins.len()];
            return;
        }

        let current_min = *self.bin_edges.first().unwrap_or(&value);
        let current_max = *self.bin_edges.last().unwrap_or(&(value + 1.0));

        if value < current_min || value > current_max {
            self.rebuild_histogram(value);
            return;
        }

        let bin_count = self.bins.len();
        let range = current_max - current_min;
        if range < 1e-6 {
            return;
        }

        let bin_idx = ((value - current_min) / range * (bin_count - 1) as f64) as usize;
        let bin_idx = bin_idx.min(bin_count - 1);
        self.bins[bin_idx] += 1.0;
    }

    fn rebuild_histogram(&mut self, new_value: f64) {
        let bin_count = self.bins.len();
        let old_min = self.bin_edges.first().copied().unwrap_or(new_value);
        let old_max = self.bin_edges.last().copied().unwrap_or(new_value + 1.0);

        let new_min = old_min.min(new_value);
        let new_max = old_max.max(new_value);
        let new_range = (new_max - new_min).max(1e-6);

        let mut new_bins = vec![0.0; bin_count];
        let mut new_edges = Vec::with_capacity(bin_count + 1);

        for i in 0..=bin_count {
            new_edges.push(new_min + (new_range * i as f64 / bin_count as f64));
        }

        if self.total_count > 0 {
            let old_total: f64 = self.bins.iter().sum();
            for i in 0..bin_count {
                let old_center = old_min + (old_max - old_min) * (i as f64 + 0.5) / bin_count as f64;
                let new_bin = ((old_center - new_min) / new_range * (bin_count - 1) as f64) as usize;
                let new_bin = new_bin.min(bin_count - 1);
                new_bins[new_bin] += self.bins[i];
            }
        }

        self.bins = new_bins;
        self.bin_edges = new_edges;
    }

    fn update_percentiles_from_moments(&mut self) {
        if self.total_count < 2 {
            return;
        }
        let std = self.variance.sqrt();
        if std < 1e-6 {
            let m = self.mean;
            self.p10 = m;
            self.p25 = m;
            self.p50 = m;
            self.p75 = m;
            self.p90 = m;
            self.p95 = m;
            self.p99 = m;
            return;
        }

        let skew = if self.total_count >= 3 {
            let n = self.total_count as f64;
            (n / ((n - 1.0) * (n - 2.0))).sqrt() * (self.skewness / (std.powi(3) * n))
        } else {
            0.0
        };

        self.p50 = self.mean;
        self.p25 = self.mean - 0.6745 * std + 0.125 * skew * std;
        self.p75 = self.mean + 0.6745 * std + 0.125 * skew * std;
        self.p10 = self.mean - 1.2816 * std + 0.2 * skew * std;
        self.p90 = self.mean + 1.2816 * std + 0.2 * skew * std;
        self.p95 = self.mean + 1.6449 * std + 0.25 * skew * std;
        self.p99 = self.mean + 2.3263 * std + 0.35 * skew * std;
    }

    pub fn std_dev(&self) -> f64 {
        if self.total_count < 2 {
            return 0.0;
        }
        (self.variance / (self.total_count as f64 - 1.0)).sqrt()
    }

    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    pub fn cdf(&self, value: f64) -> f64 {
        if self.total_count == 0 || self.bins.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = self.bins.iter().sum();
        if total_weight < f64::MIN_POSITIVE {
            return 0.0;
        }

        let bin_min = self.bin_edges.first().copied().unwrap_or(0.0);
        let bin_max = self.bin_edges.last().copied().unwrap_or(1.0);

        if value <= bin_min {
            return 0.0;
        }
        if value >= bin_max {
            return 1.0;
        }

        let range = bin_max - bin_min;
        if range < 1e-6 {
            return 0.5;
        }

        let bin_count = self.bins.len();
        let target_bin_f = (value - bin_min) / range * (bin_count - 1) as f64;
        let target_bin = target_bin_f as usize;
        let frac = target_bin_f - target_bin as f64;

        let mut cum_weight = 0.0;
        for i in 0..target_bin.min(bin_count - 1) {
            cum_weight += self.bins[i];
        }
        if target_bin < bin_count {
            cum_weight += self.bins[target_bin] * frac;
        }

        (cum_weight / total_weight).clamp(0.0, 1.0)
    }
}

impl AdaptiveTrafficLearner {
    pub fn new(learning_rate: f64, min_samples: usize) -> Self {
        AdaptiveTrafficLearner {
            observed_profiles: HashMap::new(),
            active_profile: None,
            learning_rate: learning_rate.clamp(0.001, 0.5),
            min_samples: min_samples.max(10),
            online_updater: OnlineStatisticsUpdater::new(learning_rate.clamp(0.001, 0.5)),
            similarity_checker: ProfileSimilarityChecker::new(),
            anomaly_detector: AnomalyDetector::new(DEFAULT_ANOMALY_THRESHOLD, DEFAULT_ANOMALY_WINDOW),
            raw_size_samples: HashMap::new(),
            raw_timing_samples: HashMap::new(),
            raw_burst_data: HashMap::new(),
            raw_session_durations: HashMap::new(),
            raw_idle_periods: HashMap::new(),
            raw_entropy_samples: HashMap::new(),
            direction_counts: HashMap::new(),
            current_burst: None,
            last_packet_time: None,
            session_start: None,
            total_observations: 0,
            anomaly_count: 0,
        }
    }

    pub fn observe_packet(&mut self, packet: &PacketRecord, context: &str) {
        let now = std::time::Instant::now();
        self.total_observations += 1;

        if self.session_start.is_none() {
            self.session_start = Some(now);
        }

        self.online_updater.update(packet.size as f64);

        self.raw_size_samples
            .entry(context.to_string())
            .or_insert_with(Vec::new)
            .push(packet.size as f64);

        if let Some(last_time) = self.last_packet_time {
            let iat_ms = now.duration_since(last_time).as_secs_f64() * 1000.0;
            self.raw_timing_samples
                .entry(context.to_string())
                .or_insert_with(Vec::new)
                .push(iat_ms);

            self.update_burst_tracking(packet, iat_ms, context);
        }

        self.raw_entropy_samples
            .entry(context.to_string())
            .or_insert_with(Vec::new)
            .push(packet.payload_entropy);

        let entry = self
            .direction_counts
            .entry(context.to_string())
            .or_insert((0, 0));
        match packet.direction {
            crate::masking::adversarial::PacketDirection::Up => entry.0 += 1,
            crate::masking::adversarial::PacketDirection::Down => entry.1 += 1,
        }

        self.last_packet_time = Some(now);

        if self.total_observations % self.min_samples == 0 {
            self.update_profile(context);
        }
    }

    fn update_burst_tracking(&mut self, packet: &PacketRecord, iat_ms: f64, context: &str) {
        if iat_ms > BURST_GAP_MS {
            if let Some(burst) = self.current_burst.take() {
                if burst.packet_count >= 2 {
                    self.raw_burst_data
                        .entry(context.to_string())
                        .or_insert_with(Vec::new)
                        .push(burst);
                }
            }
            self.current_burst = Some(BurstObservation {
                packet_count: 1,
                inter_packet_times: Vec::new(),
                packet_sizes: vec![packet.size as f64],
            });
        } else if let Some(ref mut burst) = self.current_burst {
            burst.packet_count += 1;
            burst.inter_packet_times.push(iat_ms);
            burst.packet_sizes.push(packet.size as f64);
        }
    }

    pub fn get_profile(&self, context: &str) -> Option<&UserProfile> {
        self.observed_profiles.get(context)
    }

    pub fn get_best_profile(&self) -> Option<&UserProfile> {
        self.observed_profiles
            .values()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn update_profile(&mut self, context: &str) {
        let size_samples = self.raw_size_samples.get(context);
        let timing_samples = self.raw_timing_samples.get(context);

        if size_samples.map_or(0, |s| s.len()) < self.min_samples {
            return;
        }

        let size_samples = size_samples.cloned().unwrap_or_default();
        let timing_samples = timing_samples.cloned().unwrap_or_default();

        let size_dist = self.build_empirical_distribution(&size_samples);
        let timing_dist = self.build_empirical_distribution(&timing_samples);

        let burst_patterns = self.detect_burst_patterns(context);

        let session_dist = self.build_empirical_distribution(
            self.raw_session_durations.get(context).map(|v| v.as_slice()).unwrap_or(&[]),
        );
        let idle_dist = self.build_empirical_distribution(
            self.raw_idle_periods.get(context).map(|v| v.as_slice()).unwrap_or(&[]),
        );

        let (up, down) = self.direction_counts.get(context).copied().unwrap_or((0, 0));
        let up_down_ratio = if down > 0 {
            up as f64 / down as f64
        } else {
            1.0
        };

        let entropy_samples = self.raw_entropy_samples.get(context);
        let payload_entropy = entropy_samples
            .filter(|s| !s.is_empty())
            .map(|s| s.iter().sum::<f64>() / s.len() as f64)
            .unwrap_or(7.0);

        let sample_count = size_samples.len();
        let existing_confidence = self
            .observed_profiles
            .get(context)
            .map(|p| p.confidence)
            .unwrap_or(0.0);

        let confidence = self.compute_confidence(sample_count, &size_dist, &timing_dist);
        let blended_confidence = if existing_confidence > 0.0 {
            existing_confidence * PROFILE_CONFIDENCE_DECAY + confidence * (1.0 - PROFILE_CONFIDENCE_DECAY)
        } else {
            confidence
        };

        let profile = UserProfile {
            name: context.to_string(),
            size_distribution: size_dist,
            timing_distribution: timing_dist,
            burst_patterns,
            session_duration_dist: session_dist,
            idle_period_dist: idle_dist,
            up_down_ratio,
            payload_entropy,
            sample_count,
            confidence: blended_confidence.clamp(0.0, 1.0),
        };

        tracing::info!(
            context = context,
            samples = sample_count,
            confidence = profile.confidence,
            burst_patterns = profile.burst_patterns.len(),
            "Updated adaptive traffic profile"
        );

        self.observed_profiles.insert(context.to_string(), profile);
        self.active_profile = Some(context.to_string());
    }

    fn build_empirical_distribution(&self, samples: &[f64]) -> EmpiricalDistribution {
        let mut dist = EmpiricalDistribution::new(DEFAULT_BIN_COUNT);
        if !samples.is_empty() {
            dist.merge_samples(samples);
        }
        dist
    }

    fn detect_burst_patterns(&self, context: &str) -> Vec<LearnedBurstPattern> {
        let burst_data = match self.raw_burst_data.get(context) {
            Some(data) if data.len() >= MIN_BURST_CLUSTER_SIZE => data,
            _ => return Vec::new(),
        };

        let mut clusters: Vec<LearnedBurstPattern> = Vec::new();

        let sorted_by_count = {
            let mut sorted = burst_data.to_vec();
            sorted.sort_by(|a, b| {
                a.packet_count
                    .partial_cmp(&b.packet_count)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted
        };

        let total_bursts = sorted_by_count.len() as f64;

        let mut current_cluster = vec![&sorted_by_count[0]];
        let mut cluster_start = sorted_by_count[0].packet_count;

        for obs in sorted_by_count.iter().skip(1) {
            let gap = (obs.packet_count as i64 - cluster_start as i64).abs();
            if gap <= 3 && current_cluster.len() < MAX_BURST_PATTERNS * 2 {
                current_cluster.push(obs);
            } else {
                if current_cluster.len() >= MIN_BURST_CLUSTER_SIZE {
                    if let Some(pattern) = self.build_burst_pattern_from_cluster(&current_cluster, total_bursts) {
                        clusters.push(pattern);
                    }
                }
                current_cluster = vec![obs];
                cluster_start = obs.packet_count;
            }
        }

        if current_cluster.len() >= MIN_BURST_CLUSTER_SIZE {
            if let Some(pattern) = self.build_burst_pattern_from_cluster(&current_cluster, total_bursts) {
                clusters.push(pattern);
            }
        }

        if clusters.len() > MAX_BURST_PATTERNS {
            clusters.sort_by(|a, b| {
                b.frequency
                    .partial_cmp(&a.frequency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            clusters.truncate(MAX_BURST_PATTERNS);
        }

        clusters
    }

    fn build_burst_pattern_from_cluster(
        &self,
        cluster: &[&BurstObservation],
        total_bursts: f64,
    ) -> Option<LearnedBurstPattern> {
        if cluster.is_empty() {
            return None;
        }

        let avg_packet_count = cluster.iter().map(|o| o.packet_count as f64).sum::<f64>() / cluster.len() as f64;

        let avg_inter_packet_time = cluster
            .iter()
            .filter_map(|o| {
                if o.inter_packet_times.is_empty() {
                    None
                } else {
                    Some(o.inter_packet_times.iter().sum::<f64>() / o.inter_packet_times.len() as f64)
                }
            })
            .sum::<f64>()
            / cluster.len() as f64;

        let avg_packet_size = cluster
            .iter()
            .filter_map(|o| {
                if o.packet_sizes.is_empty() {
                    None
                } else {
                    Some(o.packet_sizes.iter().sum::<f64>() / o.packet_sizes.len() as f64)
                }
            })
            .sum::<f64>()
            / cluster.len() as f64;

        let frequency = cluster.len() as f64 / total_bursts;
        let confidence = (cluster.len() as f64 / MIN_BURST_CLUSTER_SIZE as f64).min(1.0);

        let name = if avg_packet_count < 5.0 {
            "micro_burst".to_string()
        } else if avg_packet_count < 15.0 {
            "short_burst".to_string()
        } else if avg_packet_count < 50.0 {
            "medium_burst".to_string()
        } else {
            "long_burst".to_string()
        };

        Some(LearnedBurstPattern {
            name,
            avg_packet_count,
            avg_inter_packet_time,
            avg_packet_size,
            frequency,
            confidence,
        })
    }

    fn compute_confidence(
        &self,
        sample_count: usize,
        size_dist: &EmpiricalDistribution,
        timing_dist: &EmpiricalDistribution,
    ) -> f64 {
        let sample_factor = (sample_count as f64 / (self.min_samples * 10) as f64).min(1.0);

        let size_stability = if size_dist.total_count > 1 {
            let cv = size_dist.std_dev() / size_dist.mean.abs().max(1e-6);
            (1.0 - (cv / 3.0).min(1.0)).max(0.0)
        } else {
            0.0
        };

        let timing_stability = if timing_dist.total_count > 1 {
            let cv = timing_dist.std_dev() / timing_dist.mean.abs().max(1e-6);
            (1.0 - (cv / 3.0).min(1.0)).max(0.0)
        } else {
            0.0
        };

        (sample_factor * 0.4 + size_stability * 0.3 + timing_stability * 0.3).clamp(0.0, 1.0)
    }

    pub fn check_similarity(&self, context: &str) -> f64 {
        let profile = match self.observed_profiles.get(context) {
            Some(p) => p,
            None => return 0.0,
        };

        let size_samples = match self.raw_size_samples.get(context) {
            Some(s) if s.len() >= self.min_samples => s,
            _ => return 0.0,
        };

        let timing_samples = self.raw_timing_samples.get(context);

        let size_dist = self.build_empirical_distribution(size_samples);
        let size_similarity = self
            .similarity_checker
            .distribution_similarity(&size_dist, &profile.size_distribution);

        let timing_similarity = if let Some(ts) = timing_samples {
            if ts.len() >= self.min_samples {
                let timing_dist = self.build_empirical_distribution(ts);
                self.similarity_checker
                    .distribution_similarity(&timing_dist, &profile.timing_distribution)
            } else {
                0.5
            }
        } else {
            0.5
        };

        size_similarity * 0.6 + timing_similarity * 0.4
    }

    pub fn detect_anomaly(&mut self) -> bool {
        if let Some(active) = self.active_profile.clone() {
            let similarity = self.check_similarity(&active);
            let anomaly_score = 1.0 - similarity;
            self.anomaly_detector.add_score(anomaly_score);

            if self.anomaly_detector.is_anomalous() {
                self.anomaly_count += 1;
                tracing::warn!(
                    context = %active,
                    anomaly_score = anomaly_score,
                    threshold = self.anomaly_detector.threshold(),
                    "Anomaly detected: masking diverging from learned profile"
                );
                return true;
            }
        }
        false
    }

    pub fn generate_masked_packet(&mut self, context: &str, rng: &mut impl Rng) -> MaskedPacket {
        let profile = self.observed_profiles.get(context);

        let (size, delay_ms) = if let Some(profile) = profile {
            let size = profile.size_distribution.sample(rng).max(1.0);
            let delay_ms = profile.timing_distribution.sample(rng).max(0.0);
            (size, delay_ms)
        } else if !self.raw_size_samples.is_empty() {
            let all_sizes: Vec<f64> = self
                .raw_size_samples
                .values()
                .flat_map(|v| v.iter().copied())
                .collect();
            if all_sizes.is_empty() {
                (128.0, 50.0)
            } else {
                let idx = rng.gen_range(0..all_sizes.len());
                let size = all_sizes[idx];
                let all_timings: Vec<f64> = self
                    .raw_timing_samples
                    .values()
                    .flat_map(|v| v.iter().copied())
                    .collect();
                let delay_ms = if all_timings.is_empty() {
                    50.0
                } else {
                    all_timings[rng.gen_range(0..all_timings.len())]
                };
                (size, delay_ms)
            }
        } else {
            (128.0, 50.0)
        };

        let packet_size = size as usize;
        let mut data = vec![0u8; packet_size];
        rng.fill(&mut data[..]);

        let delay = Duration::from_millis(delay_ms as u64);

        tracing::trace!(
            context = context,
            packet_size = packet_size,
            delay_ms = delay_ms,
            "Generated masked packet from learned profile"
        );

        MaskedPacket {
            data,
            delay,
            context: context.to_string(),
        }
    }

    pub fn export_profiles(&self) -> PhantomResult<String> {
        #[derive(Serialize)]
        struct ExportableProfile {
            name: String,
            size_distribution: EmpiricalDistribution,
            timing_distribution: EmpiricalDistribution,
            burst_patterns: Vec<LearnedBurstPattern>,
            session_duration_dist: EmpiricalDistribution,
            idle_period_dist: EmpiricalDistribution,
            up_down_ratio: f64,
            payload_entropy: f64,
            sample_count: usize,
            confidence: f64,
        }

        let exportable: HashMap<String, ExportableProfile> = self
            .observed_profiles
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    ExportableProfile {
                        name: v.name.clone(),
                        size_distribution: v.size_distribution.clone(),
                        timing_distribution: v.timing_distribution.clone(),
                        burst_patterns: v.burst_patterns.clone(),
                        session_duration_dist: v.session_duration_dist.clone(),
                        idle_period_dist: v.idle_period_dist.clone(),
                        up_down_ratio: v.up_down_ratio,
                        payload_entropy: v.payload_entropy,
                        sample_count: v.sample_count,
                        confidence: v.confidence,
                    },
                )
            })
            .collect();

        serde_json::to_string_pretty(&exportable).map_err(|e| {
            tracing::error!(error = %e, "Failed to serialize profiles");
            PhantomError::SerializationError(e)
        })
    }

    pub fn import_profiles(&mut self, json: &str) -> PhantomResult<()> {
        #[derive(Deserialize)]
        struct ImportableProfile {
            name: String,
            size_distribution: EmpiricalDistribution,
            timing_distribution: EmpiricalDistribution,
            burst_patterns: Vec<LearnedBurstPattern>,
            session_duration_dist: EmpiricalDistribution,
            idle_period_dist: EmpiricalDistribution,
            up_down_ratio: f64,
            payload_entropy: f64,
            sample_count: usize,
            confidence: f64,
        }

        let imported: HashMap<String, ImportableProfile> = serde_json::from_str(json).map_err(|e| {
            tracing::error!(error = %e, "Failed to deserialize profiles");
            PhantomError::SerializationError(e)
        })?;

        let count = imported.len();
        for (key, profile) in imported {
            let user_profile = UserProfile {
                name: profile.name,
                size_distribution: profile.size_distribution,
                timing_distribution: profile.timing_distribution,
                burst_patterns: profile.burst_patterns,
                session_duration_dist: profile.session_duration_dist,
                idle_period_dist: profile.idle_period_dist,
                up_down_ratio: profile.up_down_ratio,
                payload_entropy: profile.payload_entropy,
                sample_count: profile.sample_count,
                confidence: profile.confidence,
            };
            self.observed_profiles.insert(key, user_profile);
        }

        tracing::info!(
            imported_count = count,
            total_profiles = self.observed_profiles.len(),
            "Imported learned profiles"
        );

        Ok(())
    }

    pub fn get_learning_stats(&self) -> LearningStats {
        let profile_details: Vec<ProfileDetail> = self
            .observed_profiles
            .values()
            .map(|p| ProfileDetail {
                name: p.name.clone(),
                sample_count: p.sample_count,
                confidence: p.confidence,
                size_mean: p.size_distribution.mean,
                size_std: p.size_distribution.std_dev(),
                timing_mean: p.timing_distribution.mean,
                timing_std: p.timing_distribution.std_dev(),
            })
            .collect();

        let profiles_learned = self.observed_profiles.len();
        let average_confidence = if profiles_learned > 0 {
            self.observed_profiles
                .values()
                .map(|p| p.confidence)
                .sum::<f64>()
                / profiles_learned as f64
        } else {
            0.0
        };

        let anomaly_rate = if self.total_observations > 0 {
            self.anomaly_count as f64 / self.total_observations as f64
        } else {
            0.0
        };

        LearningStats {
            total_observations: self.total_observations,
            profiles_learned,
            average_confidence,
            anomaly_rate,
            profile_details,
        }
    }

    pub fn active_profile(&self) -> Option<&str> {
        self.active_profile.as_deref()
    }

    pub fn set_active_profile(&mut self, context: &str) {
        self.active_profile = Some(context.to_string());
    }

    pub fn reset_context(&mut self, context: &str) {
        self.raw_size_samples.remove(context);
        self.raw_timing_samples.remove(context);
        self.raw_burst_data.remove(context);
        self.raw_session_durations.remove(context);
        self.raw_idle_periods.remove(context);
        self.raw_entropy_samples.remove(context);
        self.direction_counts.remove(context);
        self.observed_profiles.remove(context);
        self.current_burst = None;

        tracing::info!(context = context, "Reset learning state for context");
    }

    pub fn record_session_end(&mut self, context: &str) {
        if let Some(start) = self.session_start {
            let duration_ms = std::time::Instant::now()
                .duration_since(start)
                .as_secs_f64()
                * 1000.0;
            self.raw_session_durations
                .entry(context.to_string())
                .or_insert_with(Vec::new)
                .push(duration_ms);
        }
        self.session_start = None;

        if let Some(last) = self.last_packet_time {
            let idle_ms = std::time::Instant::now()
                .duration_since(last)
                .as_secs_f64()
                * 1000.0;
            if idle_ms > BURST_GAP_MS {
                self.raw_idle_periods
                    .entry(context.to_string())
                    .or_insert_with(Vec::new)
                    .push(idle_ms);
            }
        }
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.001, 0.5);
        self.online_updater = OnlineStatisticsUpdater::new(self.learning_rate);
    }

    pub fn min_samples(&self) -> usize {
        self.min_samples
    }

    pub fn set_anomaly_threshold(&mut self, threshold: f64) {
        self.anomaly_detector.set_threshold(threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_packet(size: usize, direction: crate::masking::adversarial::PacketDirection, entropy: f64) -> PacketRecord {
        PacketRecord {
            size,
            direction,
            timestamp: std::time::Instant::now(),
            payload_entropy: entropy,
        }
    }

    #[test]
    fn test_learner_creation() {
        let learner = AdaptiveTrafficLearner::new(0.05, 50);
        assert_eq!(learner.min_samples(), 50);
        assert_eq!(learner.learning_rate(), 0.05);
        assert!(learner.active_profile().is_none());
    }

    #[test]
    fn test_observe_packets_builds_profile() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..50 {
            let size = 100 + (i % 50);
            let dir = if i % 3 == 0 {
                crate::masking::adversarial::PacketDirection::Up
            } else {
                crate::masking::adversarial::PacketDirection::Down
            };
            learner.observe_packet(&make_packet(size, dir, 7.2), "test_context");
            std::thread::sleep(Duration::from_millis(1));
        }

        learner.update_profile("test_context");
        let profile = learner.get_profile("test_context");
        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert!(profile.sample_count >= 10);
        assert!(profile.confidence > 0.0);
        assert!(profile.size_distribution.mean > 0.0);
    }

    #[test]
    fn test_generate_masked_packet() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..30 {
            learner.observe_packet(
                &make_packet(128 + (i % 20), crate::masking::adversarial::PacketDirection::Up, 7.0),
                "youtube",
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        learner.update_profile("youtube");

        let masked = learner.generate_masked_packet("youtube", &mut rng);
        assert!(!masked.data.is_empty());
        assert_eq!(masked.context, "youtube");
    }

    #[test]
    fn test_export_import_roundtrip() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);
        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..30 {
            learner.observe_packet(
                &make_packet(200 + (i % 30), crate::masking::adversarial::PacketDirection::Down, 6.8),
                "telegram",
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        learner.update_profile("telegram");

        let json = learner.export_profiles().unwrap();
        assert!(!json.is_empty());

        let mut learner2 = AdaptiveTrafficLearner::new(0.05, 10);
        learner2.import_profiles(&json).unwrap();

        assert!(learner2.get_profile("telegram").is_some());
        let orig = learner.get_profile("telegram").unwrap();
        let imported = learner2.get_profile("telegram").unwrap();
        assert_eq!(orig.name, imported.name);
        assert_eq!(orig.sample_count, imported.sample_count);
    }

    #[test]
    fn test_learning_stats() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);

        for i in 0..20 {
            learner.observe_packet(
                &make_packet(150, crate::masking::adversarial::PacketDirection::Up, 7.5),
                "github",
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        learner.update_profile("github");

        let stats = learner.get_learning_stats();
        assert_eq!(stats.total_observations, 20);
        assert_eq!(stats.profiles_learned, 1);
        assert!(!stats.profile_details.is_empty());
        assert_eq!(stats.profile_details[0].name, "github");
    }

    #[test]
    fn test_online_statistics_welford() {
        let mut updater = OnlineStatisticsUpdater::new(0.05);
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        for v in &values {
            updater.update(*v);
        }

        assert!((updater.mean() - 30.0).abs() < 0.001);
        assert!(updater.variance() > 0.0);
        assert_eq!(updater.count(), 5);
    }

    #[test]
    fn test_empirical_distribution_sampling() {
        let mut dist = EmpiricalDistribution::new(32);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            dist.add_sample(100.0 + rng.gen_range(0.0..50.0));
        }

        assert!(dist.mean > 100.0);
        assert!(dist.mean < 150.0);
        assert!(!dist.is_empty());

        let sample = dist.sample(&mut rng);
        assert!(sample >= 100.0);
        assert!(sample <= 150.0);
    }

    #[test]
    fn test_similarity_checker_ks_test() {
        let checker = ProfileSimilarityChecker::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ks = checker.kolmogorov_smirnov(&a, &b);
        assert!(ks < 0.1);

        let c = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ks_diff = checker.kolmogorov_smirnov(&a, &c);
        assert!(ks_diff > ks);
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(3.0, 64);

        for _ in 0..20 {
            detector.add_score(0.1);
        }
        assert!(!detector.is_anomalous());

        detector.add_score(5.0);
        detector.add_score(6.0);
        detector.add_score(7.0);
    }

    #[test]
    fn test_multiple_contexts() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);

        for i in 0..20 {
            learner.observe_packet(
                &make_packet(80 + (i % 20), crate::masking::adversarial::PacketDirection::Up, 7.0),
                "youtube",
            );
            learner.observe_packet(
                &make_packet(200 + (i % 50), crate::masking::adversarial::PacketDirection::Down, 6.5),
                "telegram",
            );
            std::thread::sleep(Duration::from_millis(1));
        }

        learner.update_profile("youtube");
        learner.update_profile("telegram");

        assert!(learner.get_profile("youtube").is_some());
        assert!(learner.get_profile("telegram").is_some());

        let yt = learner.get_profile("youtube").unwrap();
        let tg = learner.get_profile("telegram").unwrap();

        assert!(yt.size_distribution.mean < tg.size_distribution.mean);
    }

    #[test]
    fn test_reset_context() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);

        for i in 0..20 {
            learner.observe_packet(
                &make_packet(100, crate::masking::adversarial::PacketDirection::Up, 7.0),
                "temp",
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        learner.update_profile("temp");
        assert!(learner.get_profile("temp").is_some());

        learner.reset_context("temp");
        assert!(learner.get_profile("temp").is_none());
    }

    #[test]
    fn test_generate_without_profile_falls_back() {
        let learner = AdaptiveTrafficLearner::new(0.05, 10);
        let mut rng = StdRng::seed_from_u64(42);

        let masked = learner.generate_masked_packet("unknown", &mut rng);
        assert!(!masked.data.is_empty());
        assert_eq!(masked.context, "unknown");
    }

    #[test]
    fn test_check_similarity_returns_value() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);

        for i in 0..30 {
            learner.observe_packet(
                &make_packet(150 + (i % 10), crate::masking::adversarial::PacketDirection::Up, 7.0),
                "check_test",
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        learner.update_profile("check_test");

        let similarity = learner.check_similarity("check_test");
        assert!(similarity >= 0.0);
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_session_recording() {
        let mut learner = AdaptiveTrafficLearner::new(0.05, 10);

        for i in 0..10 {
            learner.observe_packet(
                &make_packet(100, crate::masking::adversarial::PacketDirection::Up, 7.0),
                "session_test",
            );
            std::thread::sleep(Duration::from_millis(1));
        }

        learner.record_session_end("session_test");
        learner.update_profile("session_test");

        let profile = learner.get_profile("session_test");
        assert!(profile.is_some());
    }
}
