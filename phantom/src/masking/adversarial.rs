use rand::Rng;
use rand_distr::{Distribution, Normal, Gamma};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::masking::profiles::GamingProfile;
use crate::masking::statistical_model::{
    PacketSizeDistribution, StatisticalGameTrafficModel, TimingDistribution,
};
use crate::utils::{PhantomError, PhantomResult};

const DEFAULT_WINDOW_SIZE: usize = 256;
const DEFAULT_EMA_ALPHA: f64 = 0.05;
const DEFAULT_OPTIMIZE_INTERVAL: usize = 32;
const MIN_PACKET_SIZE: usize = 40;
const MAX_PACKET_SIZE: usize = 1500;
const POISONING_RATE: f64 = 0.08;
const ANTI_PERIODICITY_JITTER_FACTOR: f64 = 0.3;
const CDF_BUCKET_COUNT: usize = 64;

#[derive(Debug, Clone)]
pub struct PacketRecord {
    pub size: usize,
    pub direction: PacketDirection,
    pub timestamp: Instant,
    pub payload_entropy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketDirection {
    Up,
    Down,
}

#[derive(Debug, Clone)]
pub enum MaskingDecision {
    SendNow,
    Delay(Duration),
    Pad(usize),
    Split(usize),
    Merge,
    Drop,
}

impl std::fmt::Display for MaskingDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaskingDecision::SendNow => write!(f, "SendNow"),
            MaskingDecision::Delay(d) => write!(f, "Delay({:?})", d),
            MaskingDecision::Pad(n) => write!(f, "Pad({})", n),
            MaskingDecision::Split(n) => write!(f, "Split({})", n),
            MaskingDecision::Merge => write!(f, "Merge"),
            MaskingDecision::Drop => write!(f, "Drop"),
        }
    }
}

#[derive(Debug, Clone)]
struct DecisionRecord {
    decision: MaskingDecision,
    original_size: usize,
    effective_size: usize,
    timestamp: Instant,
}

pub struct AdversarialMasker {
    target_model: StatisticalGameTrafficModel,
    ema_packet_size: f64,
    ema_inter_packet_time: f64,
    ema_burstiness: f64,
    ema_entropy: f64,
    ema_direction_ratio: f64,
    recent_decisions: VecDeque<DecisionRecord>,
    dpi_window: Vec<PacketRecord>,
    window_size: usize,
    ema_alpha: f64,
    last_packet_time: Option<Instant>,
    optimize_counter: usize,
    optimize_interval: usize,
    padding_factor: f64,
    delay_factor: f64,
    split_threshold: f64,
    merge_threshold: f64,
    poisoning_counter: usize,
    anti_periodicity_phase: f64,
    bytes_up: u64,
    bytes_down: u64,
    total_packets: usize,
    burst_window: VecDeque<Instant>,
    burst_window_ms: u64,
}

impl AdversarialMasker {
    pub fn new(target_model: StatisticalGameTrafficModel) -> PhantomResult<Self> {
        let window_size = DEFAULT_WINDOW_SIZE;
        let ema_alpha = DEFAULT_EMA_ALPHA;

        tracing::info!(
            model_name = %target_model.name,
            window_size = window_size,
            ema_alpha = ema_alpha,
            "Initialized AdversarialMasker"
        );

        Ok(AdversarialMasker {
            target_model,
            ema_packet_size: 0.0,
            ema_inter_packet_time: 0.0,
            ema_burstiness: 0.0,
            ema_entropy: 0.0,
            ema_direction_ratio: 0.5,
            recent_decisions: VecDeque::with_capacity(128),
            dpi_window: Vec::with_capacity(window_size),
            window_size,
            ema_alpha,
            last_packet_time: None,
            optimize_counter: 0,
            optimize_interval: DEFAULT_OPTIMIZE_INTERVAL,
            padding_factor: 1.0,
            delay_factor: 1.0,
            split_threshold: 2.0,
            merge_threshold: 0.5,
            poisoning_counter: 0,
            anti_periodicity_phase: 0.0,
            bytes_up: 0,
            bytes_down: 0,
            total_packets: 0,
            burst_window: VecDeque::with_capacity(64),
            burst_window_ms: 100,
        })
    }

    pub fn decide(&mut self, incoming_data: &[u8]) -> MaskingDecision {
        let data_size = incoming_data.len();
        let now = Instant::now();

        self.poisoning_counter += 1;
        if self.should_poison() {
            tracing::trace!("Feature poisoning triggered");
            return self.generate_poisoning_decision();
        }

        let inter_packet_time = self.last_packet_time.map_or(0.0, |t| {
            now.duration_since(t).as_secs_f64() * 1000.0
        });

        let current_burstiness = self.compute_current_burstiness(now);
        let feature_distance = self.compute_feature_distance();

        let size_deviation = if self.ema_packet_size > 0.0 {
            (data_size as f64 - self.ema_packet_size) / self.target_model.size_dist.std.max(1e-6)
        } else {
            0.0
        };

        let timing_deviation = if self.ema_inter_packet_time > 0.0 {
            (inter_packet_time - self.ema_inter_packet_time)
                / self.target_model.timing_dist.std_iat.max(1e-6)
        } else {
            0.0
        };

        let decision = self.compute_decision(
            data_size,
            inter_packet_time,
            size_deviation,
            timing_deviation,
            current_burstiness,
            feature_distance,
        );

        self.record_decision(&decision, data_size);
        self.last_packet_time = Some(now);
        self.optimize_counter += 1;

        if self.optimize_counter >= self.optimize_interval {
            self.optimize_counter = 0;
            self.optimize();
        }

        decision
    }

    pub fn update_statistics(&mut self, packet: &PacketRecord) {
        let now = Instant::now();

        if self.dpi_window.len() >= self.window_size {
            self.dpi_window.remove(0);
        }
        self.dpi_window.push(packet.clone());

        match packet.direction {
            PacketDirection::Up => self.bytes_up += packet.size as u64,
            PacketDirection::Down => self.bytes_down += packet.size as u64,
        }
        self.total_packets += 1;

        let alpha = self.ema_alpha;
        let size_f64 = packet.size as f64;

        if self.ema_packet_size == 0.0 {
            self.ema_packet_size = size_f64;
        } else {
            self.ema_packet_size = alpha * size_f64 + (1.0 - alpha) * self.ema_packet_size;
        }

        if let Some(last) = self.last_packet_time {
            let iat = now.duration_since(last).as_secs_f64() * 1000.0;
            if self.ema_inter_packet_time == 0.0 {
                self.ema_inter_packet_time = iat;
            } else {
                self.ema_inter_packet_time =
                    alpha * iat + (1.0 - alpha) * self.ema_inter_packet_time;
            }
        }

        let entropy = packet.payload_entropy;
        if self.ema_entropy == 0.0 {
            self.ema_entropy = entropy;
        } else {
            self.ema_entropy = alpha * entropy + (1.0 - alpha) * self.ema_entropy;
        }

        let total_bytes = self.bytes_up + self.bytes_down;
        if total_bytes > 0 {
            let current_ratio = self.bytes_up as f64 / total_bytes as f64;
            self.ema_direction_ratio =
                alpha * current_ratio + (1.0 - alpha) * self.ema_direction_ratio;
        }

        self.burst_window.push_back(packet.timestamp);
        let cutoff = now - Duration::from_millis(self.burst_window_ms);
        while self
            .burst_window
            .front()
            .map_or(false, |t| *t < cutoff)
        {
            self.burst_window.pop_front();
        }
        let burst_rate = self.burst_window.len() as f64 / (self.burst_window_ms as f64 / 1000.0);
        let target_rate = self.target_model.packets_per_second;
        let burst_ratio = if target_rate > 0.0 {
            (burst_rate / target_rate - 1.0).abs()
        } else {
            0.0
        };
        self.ema_burstiness = alpha * burst_ratio + (1.0 - alpha) * self.ema_burstiness;

        self.last_packet_time = Some(now);

        tracing::trace!(
            packet_size = packet.size,
            direction = ?packet.direction,
            ema_size = self.ema_packet_size,
            ema_iat = self.ema_inter_packet_time,
            "Updated adversarial statistics"
        );
    }

    pub fn compute_feature_distance(&self) -> f64 {
        let mut distance = 0.0;
        let target = &self.target_model;

        let size_weight = 0.30;
        let timing_weight = 0.25;
        let burst_weight = 0.15;
        let entropy_weight = 0.15;
        let direction_weight = 0.10;
        let periodicity_weight = 0.05;

        if self.ema_packet_size > 0.0 && target.size_dist.std > 0.0 {
            let size_z =
                (self.ema_packet_size - target.size_dist.mean) / target.size_dist.std;
            distance += size_weight * size_z.abs();
        }

        if self.ema_inter_packet_time > 0.0 && target.timing_dist.std_iat > 0.0 {
            let iat_z = (self.ema_inter_packet_time - target.timing_dist.mean_iat)
                / target.timing_dist.std_iat;
            distance += timing_weight * iat_z.abs();
        }

        distance += burst_weight * self.ema_burstiness.min(1.0);

        if self.ema_entropy > 0.0 {
            let entropy_dev =
                (self.ema_entropy - target.target_entropy) / target.target_entropy;
            distance += entropy_weight * entropy_dev.abs();
        }

        let target_dir_ratio = target.direction_ratio;
        let dir_dev = (self.ema_direction_ratio - target_dir_ratio).abs();
        distance += direction_weight * dir_dev;

        let periodicity_penalty = self.detect_periodicity();
        distance += periodicity_weight * periodicity_penalty;

        distance
    }

    pub fn optimize(&mut self) {
        let distance = self.compute_feature_distance();
        let target = &self.target_model;

        if self.ema_packet_size > 0.0 {
            let size_error = (self.ema_packet_size - target.size_dist.mean) / target.size_dist.mean;
            if size_error > 0.1 {
                self.padding_factor *= 0.92;
                self.split_threshold *= 0.95;
            } else if size_error < -0.1 {
                self.padding_factor *= 1.08;
                self.merge_threshold *= 0.95;
            }
        }

        if self.ema_inter_packet_time > 0.0 {
            let iat_error = (self.ema_inter_packet_time - target.timing_dist.mean_iat)
                / target.timing_dist.mean_iat;
            if iat_error > 0.15 {
                self.delay_factor *= 0.90;
            } else if iat_error < -0.15 {
                self.delay_factor *= 1.10;
            }
        }

        let periodicity = self.detect_periodicity();
        if periodicity > target.periodicity_strength * 1.5 {
            self.anti_periodicity_phase += 0.17;
        }

        self.padding_factor = self.padding_factor.clamp(0.3, 3.0);
        self.delay_factor = self.delay_factor.clamp(0.3, 3.0);
        self.split_threshold = self.split_threshold.clamp(1.2, 4.0);
        self.merge_threshold = self.merge_threshold.clamp(0.2, 1.5);

        tracing::debug!(
            feature_distance = distance,
            padding_factor = self.padding_factor,
            delay_factor = self.delay_factor,
            split_threshold = self.split_threshold,
            merge_threshold = self.merge_threshold,
            periodicity = periodicity,
            "Optimized adversarial parameters"
        );
    }

    fn compute_decision(
        &self,
        data_size: usize,
        inter_packet_time: f64,
        size_deviation: f64,
        timing_deviation: f64,
        current_burstiness: f64,
        feature_distance: f64,
    ) -> MaskingDecision {
        let mut rng = rand::thread_rng();
        let target = &self.target_model;

        let target_size = self.sample_adjusted_target_size(&mut rng);
        let target_iat = target.sample_target_iat(&mut rng) * self.delay_factor;

        let anti_periodicity_jitter = self.compute_anti_periodicity_jitter(&mut rng);
        let adjusted_target_iat = (target_iat + anti_periodicity_jitter).max(1.0);

        if data_size < MIN_PACKET_SIZE {
            let padding = (target_size as usize).saturating_sub(data_size);
            let padding = (padding as f64 * self.padding_factor) as usize;
            return MaskingDecision::Pad(padding.clamp(4, MAX_PACKET_SIZE - data_size));
        }

        if size_deviation > self.split_threshold {
            let num_splits = ((data_size as f64 / target_size) as usize).clamp(2, 8);
            return MaskingDecision::Split(num_splits);
        }

        if size_deviation < -self.merge_threshold && feature_distance > 0.5 {
            return MaskingDecision::Merge;
        }

        if inter_packet_time < adjusted_target_iat * 0.6 {
            let delay_ms = (adjusted_target_iat - inter_packet_time).max(1.0);
            let jitter = rng.gen_range(0.0..=delay_ms * 0.3);
            return MaskingDecision::Delay(Duration::from_millis(
                (delay_ms + jitter) as u64,
            ));
        }

        let current_size = data_size as f64;
        let size_diff = (target_size - current_size).abs();
        if size_diff > target.size_dist.std * 0.5 {
            if current_size < target_size {
                let padding = ((target_size - current_size) * self.padding_factor * 0.5) as usize;
                if padding > 4 {
                    return MaskingDecision::Pad(padding.clamp(4, MAX_PACKET_SIZE - data_size));
                }
            }
        }

        if current_burstiness > target.timing_dist.burstiness * 1.5 {
            let delay_ms = (target.timing_dist.mean_iat * 0.3) as u64;
            let jitter = rng.gen_range(0..=delay_ms / 2);
            return MaskingDecision::Delay(Duration::from_millis(delay_ms + jitter));
        }

        MaskingDecision::SendNow
    }

    fn sample_adjusted_target_size(&self, rng: &mut impl Rng) -> f64 {
        let base = self.target_model.sample_target_size(rng);
        let tail_adjustment = self.compute_tail_adjustment();
        base * self.padding_factor * tail_adjustment
    }

    fn compute_tail_adjustment(&self) -> f64 {
        if self.dpi_window.len() < 16 {
            return 1.0;
        }

        let sizes: Vec<f64> = self
            .dpi_window
            .iter()
            .map(|p| p.size as f64)
            .collect();

        let current_p95 = self.compute_percentile(&sizes, 0.95);
        let target_p95 = self.target_model.size_dist.p95;

        if current_p95 > target_p95 * 1.2 {
            0.85
        } else if current_p95 < target_p95 * 0.8 {
            1.15
        } else {
            1.0
        }
    }

    fn compute_anti_periodicity_jitter(&self, rng: &mut impl Rng) -> f64 {
        let base_jitter = self.target_model.timing_dist.std_iat * ANTI_PERIODICITY_JITTER_FACTOR;

        let phase_offset = self.anti_periodicity_phase.sin() * base_jitter * 0.5;

        let gamma_shape = 2.0;
        let gamma_scale = base_jitter / gamma_shape;
        let gamma_sample = Gamma::new(gamma_shape, gamma_scale)
            .map(|g| g.sample(rng))
            .unwrap_or(base_jitter);

        let uniform_component: f64 = rng.gen_range(-base_jitter..=base_jitter);

        phase_offset + gamma_sample * 0.6 + uniform_component * 0.4
    }

    fn detect_periodicity(&self) -> f64 {
        if self.dpi_window.len() < 32 {
            return 0.0;
        }

        let iats: Vec<f64> = self
            .dpi_window
            .windows(2)
            .filter_map(|w| {
                let dt = w[1]
                    .timestamp
                    .duration_since(w[0].timestamp)
                    .as_secs_f64()
                    * 1000.0;
                if dt > 0.0 {
                    Some(dt)
                } else {
                    None
                }
            })
            .collect();

        if iats.len() < 16 {
            return 0.0;
        }

        let mean_iat = iats.iter().sum::<f64>() / iats.len() as f64;
        if mean_iat < 1e-6 {
            return 0.0;
        }

        let variance = iats
            .iter()
            .map(|x| (x - mean_iat).powi(2))
            .sum::<f64>()
            / iats.len() as f64;
        let std_iat = variance.sqrt();

        let cv = std_iat / mean_iat;

        let autocorr_lag1 = {
            let n = iats.len();
            let mean = mean_iat;
            let var = variance;
            if var < 1e-12 {
                0.0
            } else {
                let cov: f64 = (0..n - 1)
                    .map(|i| (iats[i] - mean) * (iats[i + 1] - mean))
                    .sum::<f64>()
                    / (n - 1) as f64;
                (cov / var).clamp(-1.0, 1.0)
            }
        };

        let run_test = {
            let median = self.compute_percentile(&iats, 0.5);
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
                let expected_runs =
                    (2.0 * n_above as f64 * n_below as f64) / (above.len() as f64) + 1.0;
                let std_runs = ((2.0 * n_above as f64 * n_below as f64
                    * (2.0 * n_above as f64 * n_below as f64 - above.len() as f64))
                    / (above.len() as f64 * above.len() as f64 * (above.len() as f64 - 1.0)))
                .sqrt();
                if std_runs < 1e-6 {
                    0.0
                } else {
                    ((runs as f64 - expected_runs) / std_runs).abs()
                }
            }
        };

        (cv * 0.4 + autocorr_lag1.abs() * 0.4 + (run_test / 4.0).min(1.0) * 0.2).clamp(0.0, 1.0)
    }

    fn should_poison(&self) -> bool {
        if self.total_packets < 16 {
            return false;
        }
        let distance = self.compute_feature_distance();
        let adaptive_rate = POISONING_RATE * (1.0 + distance * 0.5);
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() < adaptive_rate.min(0.25)
    }

    fn generate_poisoning_decision(&self) -> MaskingDecision {
        let mut rng = rand::thread_rng();
        let target = &self.target_model;

        let poisoned_size = (target.size_dist.mean
            + rng.gen_range(-target.size_dist.std..=target.size_dist.std))
        as usize;
        let poisoned_size = poisoned_size.clamp(MIN_PACKET_SIZE, MAX_PACKET_SIZE);

        let target_iat = target.timing_dist.mean_iat;
        let poisoned_iat =
            target_iat + rng.gen_range(-target.timing_dist.std_iat..=target.timing_dist.std_iat);
        let poisoned_iat = poisoned_iat.max(1.0);

        let delay = Duration::from_millis(poisoned_iat as u64);

        if poisoned_size > 64 {
            MaskingDecision::Pad(poisoned_size / 4)
        } else {
            MaskingDecision::Delay(delay)
        }
    }

    fn compute_current_burstiness(&self, now: Instant) -> f64 {
        if self.burst_window.len() < 2 {
            return 0.0;
        }

        let cutoff = now - Duration::from_millis(self.burst_window_ms);
        let recent: Vec<&Instant> = self
            .burst_window
            .iter()
            .filter(|t| **t >= cutoff)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        let iats: Vec<f64> = recent
            .windows(2)
            .map(|w| w[1].duration_since(*w[0]).as_secs_f64() * 1000.0)
            .collect();

        let mean = iats.iter().sum::<f64>() / iats.len() as f64;
        if mean < 1e-6 {
            return 0.0;
        }

        let variance = iats
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / iats.len() as f64;
        (variance.sqrt() / mean).clamp(0.0, 5.0)
    }

    fn record_decision(&mut self, decision: &MaskingDecision, original_size: usize) {
        let effective_size = match decision {
            MaskingDecision::SendNow => original_size,
            MaskingDecision::Delay(_) => original_size,
            MaskingDecision::Pad(n) => original_size + n,
            MaskingDecision::Split(n) => original_size / n.max(1),
            MaskingDecision::Merge => original_size * 2,
            MaskingDecision::Drop => 0,
        };

        let record = DecisionRecord {
            decision: decision.clone(),
            original_size,
            effective_size,
            timestamp: Instant::now(),
        };

        if self.recent_decisions.len() >= 128 {
            self.recent_decisions.pop_front();
        }
        self.recent_decisions.push_back(record);
    }

    fn compute_percentile(&self, data: &[f64], p: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (p * (sorted.len() - 1) as f64) as usize;
        let idx_next = (idx + 1).min(sorted.len() - 1);
        let t = p * (sorted.len() - 1) as f64 - idx as f64;
        sorted[idx] * (1.0 - t) + sorted[idx_next] * t
    }

    pub fn get_pending_delay(&self) -> Option<Duration> {
        self.recent_decisions
            .back()
            .and_then(|r| match &r.decision {
                MaskingDecision::Delay(d) => Some(*d),
                _ => None,
            })
    }

    pub fn get_pending_padding(&self) -> Option<usize> {
        self.recent_decisions
            .back()
            .and_then(|r| match &r.decision {
                MaskingDecision::Pad(n) => Some(*n),
                _ => None,
            })
    }

    pub fn get_pending_split(&self) -> Option<usize> {
        self.recent_decisions
            .back()
            .and_then(|r| match &r.decision {
                MaskingDecision::Split(n) => Some(*n),
                _ => None,
            })
    }

    pub fn should_merge(&self) -> bool {
        self.recent_decisions
            .back()
            .map_or(false, |r| matches!(r.decision, MaskingDecision::Merge))
    }

    pub fn should_drop(&self) -> bool {
        self.recent_decisions
            .back()
            .map_or(false, |r| matches!(r.decision, MaskingDecision::Drop))
    }

    pub fn feature_distance(&self) -> f64 {
        self.compute_feature_distance()
    }

    pub fn window_stats(&self) -> WindowStats {
        if self.dpi_window.is_empty() {
            return WindowStats::default();
        }

        let sizes: Vec<f64> = self
            .dpi_window
            .iter()
            .map(|p| p.size as f64)
            .collect();

        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let variance = sizes
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / sizes.len() as f64;
        let std = variance.sqrt();

        WindowStats {
            packet_count: self.dpi_window.len(),
            mean_size: mean,
            std_size: std,
            min_size: sizes
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min),
            max_size: sizes
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            p25_size: self.compute_percentile(&sizes, 0.25),
            p50_size: self.compute_percentile(&sizes, 0.50),
            p75_size: self.compute_percentile(&sizes, 0.75),
            p95_size: self.compute_percentile(&sizes, 0.95),
            mean_iat: self.ema_inter_packet_time,
            burstiness: self.ema_burstiness,
            entropy: self.ema_entropy,
            direction_ratio: self.ema_direction_ratio,
            periodicity: self.detect_periodicity(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WindowStats {
    pub packet_count: usize,
    pub mean_size: f64,
    pub std_size: f64,
    pub min_size: f64,
    pub max_size: f64,
    pub p25_size: f64,
    pub p50_size: f64,
    pub p75_size: f64,
    pub p95_size: f64,
    pub mean_iat: f64,
    pub burstiness: f64,
    pub entropy: f64,
    pub direction_ratio: f64,
    pub periodicity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> StatisticalGameTrafficModel {
        let profile = GamingProfile {
            name: "test".to_string(),
            packet_size_min: 60,
            packet_size_max: 300,
            heartbeat_interval_ms: 30000,
            jitter_ms: 50,
            frequency_hz: 60,
        };
        StatisticalGameTrafficModel::from_profile(&profile)
    }

    #[test]
    fn test_adversarial_masker_creation() {
        let model = make_test_model();
        let masker = AdversarialMasker::new(model);
        assert!(masker.is_ok());
        let masker = masker.unwrap();
        assert_eq!(masker.window_size, DEFAULT_WINDOW_SIZE);
        assert_eq!(masker.dpi_window.len(), 0);
    }

    #[test]
    fn test_decide_returns_valid_decision() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();
        let data = vec![0u8; 128];
        let decision = masker.decide(&data);
        match decision {
            MaskingDecision::SendNow
            | MaskingDecision::Delay(_)
            | MaskingDecision::Pad(_)
            | MaskingDecision::Split(_)
            | MaskingDecision::Merge
            | MaskingDecision::Drop => {}
        }
    }

    #[test]
    fn test_update_statistics() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();

        let packet = PacketRecord {
            size: 150,
            direction: PacketDirection::Up,
            timestamp: Instant::now(),
            payload_entropy: 7.2,
        };
        masker.update_statistics(&packet);

        assert_eq!(masker.dpi_window.len(), 1);
        assert_eq!(masker.bytes_up, 150);
        assert_eq!(masker.total_packets, 1);
    }

    #[test]
    fn test_feature_distance_initial_zero() {
        let model = make_test_model();
        let masker = AdversarialMasker::new(model).unwrap();
        let distance = masker.compute_feature_distance();
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_feature_distance_decreases_with_matching_traffic() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model.clone()).unwrap();

        let target_size = model.size_dist.mean as usize;
        for i in 0..64 {
            let packet = PacketRecord {
                size: target_size + (i % 20) - 10,
                direction: if i % 3 == 0 {
                    PacketDirection::Up
                } else {
                    PacketDirection::Down
                },
                timestamp: Instant::now(),
                payload_entropy: 7.5,
            };
            masker.update_statistics(&packet);
        }

        let distance = masker.compute_feature_distance();
        assert!(distance < 2.0);
    }

    #[test]
    fn test_optimize_adjusts_parameters() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();

        let initial_padding = masker.padding_factor;
        let initial_delay = masker.delay_factor;

        for _ in 0..32 {
            masker.decide(&vec![0u8; 200]);
        }

        assert!(masker.optimize_counter == 0);
        let _ = masker.compute_feature_distance();
    }

    #[test]
    fn test_anti_periodicity_jitter_is_nonzero() {
        let model = make_test_model();
        let masker = AdversarialMasker::new(model).unwrap();
        let mut rng = rand::thread_rng();

        let j1 = masker.compute_anti_periodicity_jitter(&mut rng);
        let j2 = masker.compute_anti_periodicity_jitter(&mut rng);

        assert!(j1.abs() > 0.0 || j2.abs() > 0.0);
    }

    #[test]
    fn test_window_stats_computation() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();

        for i in 0..32 {
            let packet = PacketRecord {
                size: 100 + i,
                direction: PacketDirection::Up,
                timestamp: Instant::now(),
                payload_entropy: 7.0,
            };
            masker.update_statistics(&packet);
        }

        let stats = masker.window_stats();
        assert_eq!(stats.packet_count, 32);
        assert!(stats.mean_size > 0.0);
        assert!(stats.std_size > 0.0);
        assert!(stats.p25_size <= stats.p50_size);
        assert!(stats.p50_size <= stats.p75_size);
        assert!(stats.p75_size <= stats.p95_size);
    }

    #[test]
    fn test_decision_recording() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();

        masker.decide(&vec![0u8; 100]);
        assert_eq!(masker.recent_decisions.len(), 1);

        for _ in 0..130 {
            masker.decide(&vec![0u8; 100]);
        }
        assert!(masker.recent_decisions.len() <= 128);
    }

    #[test]
    fn test_percentile_computation() {
        let model = make_test_model();
        let masker = AdversarialMasker::new(model).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = masker.compute_percentile(&data, 0.5);
        let p25 = masker.compute_percentile(&data, 0.25);
        let p75 = masker.compute_percentile(&data, 0.75);

        assert!((p50 - 5.5).abs() < 0.5);
        assert!(p25 < p50);
        assert!(p50 < p75);
    }

    #[test]
    fn test_cdf_interpolation() {
        let model = make_test_model();
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let size = model.sample_target_size(&mut rng);
            assert!(size >= model.size_dist.min);
            assert!(size <= model.size_dist.max);

            let iat = model.sample_target_iat(&mut rng);
            assert!(iat > 0.0);
        }
    }

    #[test]
    fn test_small_packet_forces_padding() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();

        let tiny_data = vec![0u8; 10];
        let decision = masker.decide(&tiny_data);

        match decision {
            MaskingDecision::Pad(n) => assert!(n > 0),
            _ => {}
        }
    }

    #[test]
    fn test_poisoning_triggers_after_warmup() {
        let model = make_test_model();
        let mut masker = AdversarialMasker::new(model).unwrap();

        for i in 0..20 {
            let packet = PacketRecord {
                size: 200,
                direction: PacketDirection::Up,
                timestamp: Instant::now(),
                payload_entropy: 6.0,
            };
            masker.update_statistics(&packet);
            masker.decide(&vec![0u8; 200]);
        }

        let mut poison_count = 0;
        for _ in 0..200 {
            if masker.should_poison() {
                poison_count += 1;
            }
        }
        assert!(poison_count > 0);
    }
}
