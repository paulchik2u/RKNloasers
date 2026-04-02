use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::collections::VecDeque;
use std::time::Duration;

use crate::masking::statistical_model::StatisticalGameTrafficModel;
use crate::utils::{PhantomError, PhantomResult};

#[derive(Debug, Clone)]
pub struct Oscillator {
    frequency: f64,
    amplitude: f64,
    phase: f64,
    drift: f64,
}

impl Oscillator {
    fn new(frequency: f64, amplitude: f64, phase: f64, drift: f64) -> Self {
        Self {
            frequency,
            amplitude,
            phase,
            drift,
        }
    }

    fn tick(&mut self, dt: f64) -> f64 {
        self.phase += self.frequency * dt;
        self.frequency += self.drift * dt;
        self.amplitude * (self.phase * std::f64::consts::TAU).sin()
    }

    fn value(&self) -> f64 {
        self.amplitude * (self.phase * std::f64::consts::TAU).sin()
    }
}

#[derive(Debug, Clone)]
pub struct PulseGenerator {
    oscillators: Vec<Oscillator>,
}

impl PulseGenerator {
    fn new(base_frequency: f64, periodicity_strength: f64) -> Self {
        let primary_amplitude = periodicity_strength * 1000.0 / base_frequency;
        let secondary_amplitude = primary_amplitude * 0.3;
        let tertiary_amplitude = primary_amplitude * 0.1;

        let oscillators = vec![
            Oscillator::new(
                base_frequency,
                primary_amplitude,
                0.0,
                base_frequency * 0.0001,
            ),
            Oscillator::new(
                base_frequency * 2.0,
                secondary_amplitude,
                std::f64::consts::FRAC_PI_4,
                base_frequency * 0.00005,
            ),
            Oscillator::new(
                base_frequency * 0.5,
                tertiary_amplitude,
                std::f64::consts::FRAC_PI_2,
                base_frequency * 0.00002,
            ),
        ];

        tracing::debug!(
            base_frequency = base_frequency,
            oscillators = oscillators.len(),
            "PulseGenerator initialized"
        );

        Self { oscillators }
    }

    fn tick(&mut self, dt: f64) -> f64 {
        let mut total = 0.0;
        for osc in &mut self.oscillators {
            total += osc.tick(dt);
        }
        total
    }

    fn value(&self) -> f64 {
        self.oscillators.iter().map(|o| o.value()).sum()
    }

    fn recalibrate(&mut self, base_frequency: f64, periodicity_strength: f64) {
        let primary_amplitude = periodicity_strength * 1000.0 / base_frequency;

        if let Some(osc) = self.oscillators.get_mut(0) {
            osc.frequency = base_frequency;
            osc.amplitude = primary_amplitude;
            osc.drift = base_frequency * 0.0001;
        }
        if let Some(osc) = self.oscillators.get_mut(1) {
            osc.frequency = base_frequency * 2.0;
            osc.amplitude = primary_amplitude * 0.3;
            osc.drift = base_frequency * 0.00005;
        }
        if let Some(osc) = self.oscillators.get_mut(2) {
            osc.frequency = base_frequency * 0.5;
            osc.amplitude = primary_amplitude * 0.1;
            osc.drift = base_frequency * 0.00002;
        }

        tracing::debug!(
            base_frequency = base_frequency,
            "PulseGenerator recalibrated"
        );
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveTimingController {
    target_median: f64,
    target_cv: f64,
    current_median: f64,
    current_cv: f64,
    kp: f64,
    ki: f64,
    kd: f64,
    error_integral: f64,
    prev_error: f64,
}

impl AdaptiveTimingController {
    fn new(target_median: f64, target_cv: f64) -> Self {
        let kp = 0.01;
        let ki = 0.001;
        let kd = 0.005;

        tracing::debug!(
            target_median_ms = target_median,
            target_cv = target_cv,
            kp = kp,
            ki = ki,
            kd = kd,
            "AdaptiveTimingController initialized"
        );

        Self {
            target_median,
            target_cv,
            current_median: target_median,
            current_cv: target_cv,
            kp,
            ki,
            kd,
            error_integral: 0.0,
            prev_error: 0.0,
        }
    }

    fn update(&mut self, measured_median: f64, measured_cv: f64) -> f64 {
        self.current_median = measured_median;
        self.current_cv = measured_cv;

        let median_error = self.target_median - measured_median;
        let cv_error = self.target_cv - measured_cv;
        let combined_error = median_error * 0.7 + cv_error * self.target_median * 0.3;

        self.error_integral += combined_error;
        let error_integral_clamped = self.error_integral.clamp(-100.0, 100.0);

        let derivative = combined_error - self.prev_error;
        self.prev_error = combined_error;

        let correction = self.kp * combined_error
            + self.ki * error_integral_clamped
            + self.kd * derivative;

        correction.clamp(-5.0, 5.0)
    }

    fn recalibrate(&mut self, target_median: f64, target_cv: f64) {
        self.target_median = target_median;
        self.target_cv = target_cv;
        self.error_integral = 0.0;
        self.prev_error = 0.0;

        tracing::debug!(
            new_target_median_ms = target_median,
            new_target_cv = target_cv,
            "AdaptiveTimingController recalibrated"
        );
    }

    fn adjustment_strength(&self) -> f64 {
        let median_ratio = if self.target_median > 0.0 {
            (self.current_median / self.target_median).clamp(0.5, 2.0)
        } else {
            1.0
        };
        let cv_ratio = if self.target_cv > 0.0 {
            (self.current_cv / self.target_cv).clamp(0.5, 2.0)
        } else {
            1.0
        };
        (median_ratio + cv_ratio) / 2.0
    }
}

#[derive(Debug, Clone)]
pub struct PacketTimingSimulator {
    game_model: StatisticalGameTrafficModel,
    phase: f64,
    tick_rate: f64,
    phase_noise: f64,
    pulse_generator: PulseGenerator,
    adaptive_controller: AdaptiveTimingController,
    timing_history: VecDeque<f64>,
    history_size: usize,
}

#[derive(Debug, Clone)]
pub struct TimingValidation {
    pub chi_squared: f64,
    pub ks_statistic: f64,
    pub passes: bool,
}

#[derive(Debug, Clone)]
pub struct TimingStatistics {
    pub median: f64,
    pub mean: f64,
    pub std: f64,
    pub cv: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
    pub sample_count: usize,
}

impl PacketTimingSimulator {
    pub fn new(game_model: StatisticalGameTrafficModel) -> Self {
        let tick_rate = game_model.packets_per_second.max(1.0);
        let base_period_ms = 1000.0 / tick_rate;

        let target_median = game_model.timing_dist.p50_iat;
        let target_cv = game_model.timing_dist.cv;

        let periodicity_strength = game_model.periodicity_strength;
        let pulse_generator = PulseGenerator::new(tick_rate, periodicity_strength);

        let adaptive_controller = AdaptiveTimingController::new(target_median, target_cv);

        let phase_noise = game_model.timing_dist.std_iat * 0.1;

        let history_size = 256;
        let timing_history = VecDeque::with_capacity(history_size);

        tracing::info!(
            game = %game_model.name,
            tick_rate_hz = tick_rate,
            base_period_ms = base_period_ms,
            target_median_ms = target_median,
            target_cv = target_cv,
            phase_noise_ms = phase_noise,
            "PacketTimingSimulator initialized"
        );

        Self {
            game_model,
            phase: 0.0,
            tick_rate,
            phase_noise,
            pulse_generator,
            adaptive_controller,
            timing_history,
            history_size,
        }
    }

    pub fn next_delay(&mut self, rng: &mut impl Rng) -> Duration {
        let base_period_ms = 1000.0 / self.tick_rate;

        let pulse_contribution = self.pulse_generator.tick(1.0 / self.tick_rate);

        let phase_step = self.tick_rate / self.tick_rate;
        self.phase += phase_step;
        let phase_offset = (self.phase * std::f64::consts::TAU).sin() * self.phase_noise;

        let model_sample = self.game_model.sample_target_iat(rng);

        let noise_dist = Normal::new(0.0, self.phase_noise)
            .unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("fallback normal distribution"));
        let gaussian_noise = noise_dist.sample(rng);

        let burst_adjustment = self.apply_burst_timing(rng);

        let pid_correction = self.adaptive_controller.adjustment_strength();

        let raw_delay = model_sample
            + pulse_contribution * 0.2
            + phase_offset * 0.3
            + gaussian_noise * 0.15
            + burst_adjustment
            + (base_period_ms - model_sample) * pid_correction * 0.1;

        let delay_ms = raw_delay.max(0.5);

        let delay = Duration::from_micros((delay_ms * 1000.0) as u64);

        tracing::trace!(
            game = %self.game_model.name,
            delay_ms = delay_ms,
            base_period_ms = base_period_ms,
            pulse_ms = pulse_contribution,
            phase_offset_ms = phase_offset,
            burst_adjust_ms = burst_adjustment,
            pid_factor = pid_correction,
            "Generated inter-packet delay"
        );

        delay
    }

    pub fn update(&mut self, actual_delay: Duration) {
        let actual_ms = actual_delay.as_secs_f64() * 1000.0;

        if self.timing_history.len() >= self.history_size {
            self.timing_history.pop_front();
        }
        self.timing_history.push_back(actual_ms);

        if self.timing_history.len() >= 32 {
            let stats = self.compute_history_statistics();

            let correction = self
                .adaptive_controller
                .update(stats.median, stats.cv);

            if correction.abs() > 0.5 {
                self.phase_noise *= 1.0 + correction * 0.01;
                self.phase_noise = self.phase_noise.clamp(0.1, 50.0);
            }
        }

        if self.timing_history.len() == self.history_size
            && self.timing_history.len() % 64 == 0
        {
            let validation = self.validate_timing();
            if !validation.passes {
                tracing::warn!(
                    game = %self.game_model.name,
                    chi_squared = validation.chi_squared,
                    ks_statistic = validation.ks_statistic,
                    "Timing validation failed, triggering calibration"
                );
                self.calibrate();
            }
        }
    }

    pub fn validate_timing(&self) -> TimingValidation {
        if self.timing_history.len() < 30 {
            return TimingValidation {
                chi_squared: f64::INFINITY,
                ks_statistic: f64::INFINITY,
                passes: false,
            };
        }

        let samples: Vec<f64> = self.timing_history.iter().copied().collect();

        let chi_squared = self.compute_chi_squared(&samples);
        let ks_statistic = self.compute_ks_statistic(&samples);

        let chi_critical = Self::chi_squared_critical_value(9, 0.05);
        let ks_critical = 1.36 / (samples.len() as f64).sqrt();

        let passes = chi_squared < chi_critical && ks_statistic < ks_critical;

        tracing::debug!(
            game = %self.game_model.name,
            chi_squared = chi_squared,
            chi_critical = chi_critical,
            ks_statistic = ks_statistic,
            ks_critical = ks_critical,
            passes = passes,
            sample_count = samples.len(),
            "Timing validation completed"
        );

        TimingValidation {
            chi_squared,
            ks_statistic,
            passes,
        }
    }

    pub fn calibrate(&mut self) {
        let target_median = self.game_model.timing_dist.p50_iat;
        let target_cv = self.game_model.timing_dist.cv;
        let target_std = self.game_model.timing_dist.std_iat;

        self.adaptive_controller.recalibrate(target_median, target_cv);

        self.pulse_generator
            .recalibrate(self.tick_rate, self.game_model.periodicity_strength);

        self.phase_noise = target_std * 0.1;
        self.phase_noise = self.phase_noise.clamp(0.1, 50.0);

        if self.timing_history.len() >= 16 {
            let stats = self.compute_history_statistics();
            let median_ratio = if stats.median > 0.0 {
                target_median / stats.median
            } else {
                1.0
            };
            self.tick_rate *= median_ratio;
            self.tick_rate = self.tick_rate.clamp(1.0, 128.0);

            tracing::info!(
                game = %self.game_model.name,
                old_tick_rate = self.tick_rate / median_ratio,
                new_tick_rate = self.tick_rate,
                "Tick rate adjusted during calibration"
            );
        }

        self.timing_history.clear();

        tracing::info!(
            game = %self.game_model.name,
            target_median_ms = target_median,
            target_cv = target_cv,
            phase_noise_ms = self.phase_noise,
            "PacketTimingSimulator calibrated"
        );
    }

    pub fn get_statistics(&self) -> TimingStatistics {
        if self.timing_history.is_empty() {
            return TimingStatistics {
                median: self.game_model.timing_dist.p50_iat,
                mean: self.game_model.timing_dist.mean_iat,
                std: self.game_model.timing_dist.std_iat,
                cv: self.game_model.timing_dist.cv,
                p25: self.game_model.timing_dist.p50_iat * 0.75,
                p75: self.game_model.timing_dist.p50_iat * 1.25,
                p95: self.game_model.timing_dist.p95_iat,
                sample_count: 0,
            };
        }

        self.compute_history_statistics()
    }

    fn apply_burst_timing(&self, rng: &mut impl Rng) -> f64 {
        if let Some(pattern) = self.game_model.should_burst(rng) {
            let ipt_range = pattern.inter_packet_time_range;
            let burst_ipt = rng.gen_range(ipt_range.0..=ipt_range.1);
            let model_iat = self.game_model.timing_dist.mean_iat;
            burst_ipt - model_iat
        } else {
            0.0
        }
    }

    fn compute_history_statistics(&self) -> TimingStatistics {
        if self.timing_history.is_empty() {
            return TimingStatistics {
                median: 0.0,
                mean: 0.0,
                std: 0.0,
                cv: 0.0,
                p25: 0.0,
                p75: 0.0,
                p95: 0.0,
                sample_count: 0,
            };
        }

        let mut sorted: Vec<f64> = self.timing_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let mean: f64 = sorted.iter().sum::<f64>() / n as f64;

        let variance: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        let cv = if mean > 0.0 { std / mean } else { 0.0 };

        let percentile = |p: f64| -> f64 {
            let idx = (p * (n - 1) as f64).round() as usize;
            sorted[idx.min(n - 1)]
        };

        TimingStatistics {
            median: percentile(0.5),
            mean,
            std,
            cv,
            p25: percentile(0.25),
            p75: percentile(0.75),
            p95: percentile(0.95),
            sample_count: n,
        }
    }

    fn compute_chi_squared(&self, samples: &[f64]) -> f64 {
        let num_bins = 10;
        let min_val = samples
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_val = samples
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val) < 1e-6 {
            return 0.0;
        }

        let bin_width = (max_val - min_val) / num_bins as f64;
        let mut observed = vec![0usize; num_bins];
        let mut expected = vec![0.0f64; num_bins];

        for &sample in samples {
            let bin = ((sample - min_val) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);
            observed[bin] += 1;
        }

        let total_samples = samples.len() as f64;
        let target_mean = self.game_model.timing_dist.mean_iat;
        let target_std = self.game_model.timing_dist.std_iat;

        let normal = match Normal::new(target_mean, target_std.max(0.01)) {
            Ok(n) => n,
            Err(_) => return f64::INFINITY,
        };

        for i in 0..num_bins {
            let bin_lower = min_val + i as f64 * bin_width;
            let bin_upper = bin_lower + bin_width;

            let cdf_upper = normal.cdf(bin_upper);
            let cdf_lower = normal.cdf(bin_lower);
            let prob = (cdf_upper - cdf_lower).max(0.0);

            expected[i] = prob * total_samples;
        }

        let mut chi_sq = 0.0;
        for i in 0..num_bins {
            if expected[i] > 0.0 {
                let diff = observed[i] as f64 - expected[i];
                chi_sq += (diff * diff) / expected[i];
            }
        }

        chi_sq
    }

    fn compute_ks_statistic(&self, samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return f64::INFINITY;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len() as f64;
        let target_mean = self.game_model.timing_dist.mean_iat;
        let target_std = self.game_model.timing_dist.std_iat.max(0.01);

        let normal = match Normal::new(target_mean, target_std) {
            Ok(n) => n,
            Err(_) => return f64::INFINITY,
        };

        let mut max_diff = 0.0;
        for (i, &value) in sorted.iter().enumerate() {
            let empirical_cdf = (i as f64 + 1.0) / n;
            let theoretical_cdf = normal.cdf(value);
            let diff = (empirical_cdf - theoretical_cdf).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        max_diff
    }

    fn chi_squared_critical_value(dof: usize, alpha: f64) -> f64 {
        match dof {
            1 => 3.841,
            2 => 5.991,
            3 => 7.815,
            4 => 9.488,
            5 => 11.070,
            6 => 12.592,
            7 => 14.067,
            8 => 15.507,
            9 => 16.919,
            10 => 18.307,
            _ => {
                let dof_f = dof as f64;
                let z = match alpha {
                    0.05 => 1.645,
                    0.01 => 2.326,
                    _ => 1.96,
                };
                dof_f * (1.0 - 2.0 / (9.0 * dof_f) + z * (2.0 / (9.0 * dof_f)).sqrt()).powi(3)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_model() -> StatisticalGameTrafficModel {
        use crate::masking::profiles::GamingProfile;
        use crate::masking::profiles::{
            BurstPatternDef, EntropyProfileDef, GaussianComponentDef, TimingBinDef,
        };

        let profile = GamingProfile {
            name: "test".to_string(),
            packet_size_min: 60,
            packet_size_max: 400,
            heartbeat_interval_ms: 30000,
            jitter_ms: 50,
            frequency_hz: 64,
            size_distribution: vec![
                GaussianComponentDef {
                    mean: 90.0,
                    std_dev: 15.0,
                    weight: 0.50,
                },
                GaussianComponentDef {
                    mean: 200.0,
                    std_dev: 50.0,
                    weight: 0.30,
                },
                GaussianComponentDef {
                    mean: 400.0,
                    std_dev: 80.0,
                    weight: 0.20,
                },
            ],
            timing_distribution: vec![
                TimingBinDef {
                    time_ms: 15.63,
                    probability: 0.65,
                },
                TimingBinDef {
                    time_ms: 31.25,
                    probability: 0.20,
                },
                TimingBinDef {
                    time_ms: 50.0,
                    probability: 0.15,
                },
            ],
            burst_patterns: vec![BurstPatternDef {
                name: "test_burst".to_string(),
                packet_count_range: (3, 7),
                inter_packet_time_range: (5.0, 15.0),
                size_range: (80, 200),
                probability: 0.10,
            }],
            client_server_size_ratio: (0.5, 1.0),
            entropy_profile: Some(EntropyProfileDef {
                mean_entropy: 7.2,
                entropy_std: 0.3,
            }),
        };

        StatisticalGameTrafficModel::from_profile(&profile).unwrap()
    }

    #[test]
    fn test_simulator_creation() {
        let model = test_model();
        let simulator = PacketTimingSimulator::new(model);
        assert_eq!(simulator.tick_rate, 64.0);
        assert_eq!(simulator.history_size, 256);
        assert!(simulator.timing_history.is_empty());
    }

    #[test]
    fn test_next_delay_produces_valid_durations() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let delay = simulator.next_delay(&mut rng);
            assert!(delay.as_micros() > 0, "Delay must be positive");
            assert!(
                delay.as_secs() < 10,
                "Delay must be less than 10 seconds"
            );
        }
    }

    #[test]
    fn test_update_records_history() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);

        for i in 0..50 {
            simulator.update(Duration::from_millis(15 + (i % 5) as u64));
        }

        assert_eq!(simulator.timing_history.len(), 50);
    }

    #[test]
    fn test_history_wraps_at_capacity() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);

        for i in 0..300 {
            simulator.update(Duration::from_millis(15 + (i % 10) as u64));
        }

        assert_eq!(simulator.timing_history.len(), simulator.history_size);
    }

    #[test]
    fn test_get_statistics_returns_valid_values() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let delay = simulator.next_delay(&mut rng);
            simulator.update(delay);
        }

        let stats = simulator.get_statistics();
        assert!(stats.sample_count > 0);
        assert!(stats.median > 0.0);
        assert!(stats.mean > 0.0);
        assert!(stats.std >= 0.0);
        assert!(stats.cv >= 0.0);
        assert!(stats.p25 <= stats.median);
        assert!(stats.median <= stats.p75);
        assert!(stats.p75 <= stats.p95);
    }

    #[test]
    fn test_validate_timing_with_sufficient_samples() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let delay = simulator.next_delay(&mut rng);
            simulator.update(delay);
        }

        let validation = simulator.validate_timing();
        assert!(validation.chi_squared.is_finite());
        assert!(validation.ks_statistic.is_finite());
    }

    #[test]
    fn test_validate_timing_with_insufficient_samples() {
        let model = test_model();
        let simulator = PacketTimingSimulator::new(model);

        let validation = simulator.validate_timing();
        assert!(!validation.passes);
    }

    #[test]
    fn test_calibration_resets_state() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..50 {
            let delay = simulator.next_delay(&mut rng);
            simulator.update(delay);
        }

        let stats_before = simulator.get_statistics();
        simulator.calibrate();

        assert!(simulator.timing_history.is_empty());
        let stats_after = simulator.get_statistics();
        assert_eq!(stats_after.sample_count, 0);
    }

    #[test]
    fn test_timing_statistics_percentile_ordering() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..200 {
            let delay = simulator.next_delay(&mut rng);
            simulator.update(delay);
        }

        let stats = simulator.get_statistics();
        assert!(
            stats.p25 <= stats.median,
            "P25 should be <= median: {} vs {}",
            stats.p25,
            stats.median
        );
        assert!(
            stats.median <= stats.p75,
            "Median should be <= P75: {} vs {}",
            stats.median,
            stats.p75
        );
        assert!(
            stats.p75 <= stats.p95,
            "P75 should be <= P95: {} vs {}",
            stats.p75,
            stats.p95
        );
    }

    #[test]
    fn test_oscillator_tick_produces_output() {
        let mut osc = Oscillator::new(1.0, 10.0, 0.0, 0.001);
        let val1 = osc.tick(0.01);
        let val2 = osc.tick(0.01);
        assert!(val1.is_finite());
        assert!(val2.is_finite());
    }

    #[test]
    fn test_pulse_generator_multiple_oscillators() {
        let mut pg = PulseGenerator::new(64.0, 0.15);
        assert_eq!(pg.oscillators.len(), 3);

        let val = pg.tick(0.01);
        assert!(val.is_finite());
    }

    #[test]
    fn test_adaptive_controller_correction() {
        let mut controller = AdaptiveTimingController::new(15.63, 0.3);

        let correction = controller.update(20.0, 0.5);
        assert!(correction.is_finite());

        let correction2 = controller.update(10.0, 0.2);
        assert!(correction2.is_finite());
    }

    #[test]
    fn test_simulator_produces_realistic_delays_for_64hz() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        let mut delays: Vec<f64> = Vec::with_capacity(500);
        for _ in 0..500 {
            let delay = simulator.next_delay(&mut rng);
            delays.push(delay.as_secs_f64() * 1000.0);
            simulator.update(delay);
        }

        let mean: f64 = delays.iter().sum::<f64>() / delays.len() as f64;
        assert!(
            mean > 5.0 && mean < 60.0,
            "Mean delay {:.2}ms out of realistic range for 64Hz",
            mean
        );
    }

    #[test]
    fn test_simulator_no_zero_delays() {
        let model = test_model();
        let mut simulator = PacketTimingSimulator::new(model);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let delay = simulator.next_delay(&mut rng);
            assert!(
                delay.as_micros() > 0,
                "Zero delay detected, which would cause tight loop"
            );
        }
    }
}
