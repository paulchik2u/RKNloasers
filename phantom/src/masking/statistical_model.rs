use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;
use std::sync::RwLock;

use crate::masking::profiles::{
    BurstPatternDef, EntropyProfileDef, GamingProfile, GaussianComponentDef, TimingBinDef,
};
use crate::utils::{PhantomError, PhantomResult};

#[derive(Debug, Clone)]
pub struct GaussianComponent {
    pub mean: f64,
    pub std_dev: f64,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct BurstPattern {
    pub name: String,
    pub packet_count_range: (usize, usize),
    pub inter_packet_time_range: (f64, f64),
    pub size_range: (usize, usize),
    pub probability: f64,
}

#[derive(Debug, Clone)]
pub struct EntropyProfile {
    pub mean_entropy: f64,
    pub entropy_std: f64,
}

#[derive(Debug, Clone)]
pub struct PacketSizeDistribution {
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone)]
pub struct TimingDistribution {
    pub mean_iat: f64,
    pub std_iat: f64,
    pub cv: f64,
    pub p50_iat: f64,
    pub p95_iat: f64,
    pub burstiness: f64,
}

#[derive(Debug, Clone)]
pub struct StatisticalGameTrafficModel {
    pub name: String,
    pub size_dist: PacketSizeDistribution,
    pub timing_dist: TimingDistribution,
    pub direction_ratio: f64,
    pub target_entropy: f64,
    pub flow_duration_mean: f64,
    pub packets_per_second: f64,
    pub periodicity_strength: f64,
    pub size_cdf: Vec<f64>,
    pub iat_cdf: Vec<f64>,
    size_distribution: Vec<GaussianComponent>,
    size_weights_cdf: Vec<f64>,
    timing_distribution: Vec<(f64, f64)>,
    timing_cdf: Vec<f64>,
    burst_patterns: Vec<BurstPattern>,
    client_server_size_ratio: (f64, f64),
    entropy_profile: EntropyProfile,
}

impl StatisticalGameTrafficModel {
    pub fn from_profile(profile: &GamingProfile) -> PhantomResult<Self> {
        tracing::info!(game = %profile.name, "Building statistical traffic model");

        let size_distribution = Self::build_size_distribution(profile)?;
        let size_weights_cdf =
            Self::compute_cdf(&size_distribution.iter().map(|c| c.weight).collect::<Vec<_>>());

        let timing_distribution: Vec<(f64, f64)> = profile
            .timing_distribution
            .iter()
            .map(|bin| (bin.time_ms, bin.probability))
            .collect();
        let timing_probs: Vec<f64> = timing_distribution.iter().map(|(_, p)| *p).collect();
        let timing_cdf = Self::compute_cdf(&timing_probs);

        let burst_patterns = profile
            .burst_patterns
            .iter()
            .map(|def| BurstPattern {
                name: def.name.clone(),
                packet_count_range: def.packet_count_range,
                inter_packet_time_range: def.inter_packet_time_range,
                size_range: def.size_range,
                probability: def.probability,
            })
            .collect();

        let entropy_profile = profile
            .entropy_profile
            .as_ref()
            .map(|ep| EntropyProfile {
                mean_entropy: ep.mean_entropy,
                entropy_std: ep.entropy_std,
            })
            .unwrap_or_else(|| EntropyProfile {
                mean_entropy: 7.0,
                entropy_std: 0.5,
            });

        let size_dist = Self::compute_size_distribution_stats(&size_distribution);
        let timing_dist = Self::compute_timing_distribution_stats(&timing_distribution);

        let size_cdf = Self::build_empirical_cdf_from_mixture(&size_distribution, 64);
        let iat_cdf = Self::build_empirical_cdf_from_timing(&timing_distribution, 64);

        let direction_ratio = profile.client_server_size_ratio.0 / profile.client_server_size_ratio.1;
        let periodicity_strength = 0.15 + (profile.jitter_ms as f64 / 200.0).min(0.3);
        let packets_per_second = profile.frequency_hz as f64;

        tracing::info!(
            game = %profile.name,
            size_components = size_distribution.len(),
            timing_bins = timing_distribution.len(),
            burst_patterns = burst_patterns.len(),
            "Statistical model built successfully"
        );

        Ok(Self {
            name: profile.name.clone(),
            size_dist,
            timing_dist,
            direction_ratio,
            target_entropy: entropy_profile.mean_entropy,
            flow_duration_mean: 300.0,
            packets_per_second,
            periodicity_strength,
            size_cdf,
            iat_cdf,
            size_distribution,
            size_weights_cdf,
            timing_distribution,
            timing_cdf,
            burst_patterns,
            client_server_size_ratio: profile.client_server_size_ratio,
            entropy_profile,
        })
    }

    pub fn from_profile_arc(profile: &GamingProfile) -> PhantomResult<Arc<RwLock<Self>>> {
        let model = Self::from_profile(profile)?;
        Ok(Arc::new(RwLock::new(model)))
    }

    pub fn sample_packet_size(&self, rng: &mut impl Rng) -> usize {
        let component_idx = self.sample_from_cdf(&self.size_weights_cdf, rng);
        let component = &self.size_distribution[component_idx];

        let normal = match Normal::new(component.mean, component.std_dev) {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    game = %self.name,
                    mean = component.mean,
                    std = component.std_dev,
                    "Failed to create normal distribution, falling back to mean"
                );
                return component.mean.round() as usize;
            }
        };

        let raw = normal.sample(rng);
        let clamped = raw.clamp(40.0, 1500.0);
        clamped.round() as usize
    }

    pub fn sample_inter_packet_time(&self, rng: &mut impl Rng) -> f64 {
        if self.timing_distribution.is_empty() {
            tracing::warn!(game = %self.name, "Empty timing distribution, using default");
            return 50.0;
        }

        let idx = self.sample_from_cdf(&self.timing_cdf, rng);
        let (base_time, _) = self.timing_distribution[idx];

        let jitter = rng.gen_range(-2.0..=2.0);
        let jittered = (base_time + jitter).max(1.0);

        jittered
    }

    pub fn should_burst(&self, rng: &mut impl Rng) -> Option<&BurstPattern> {
        let roll: f64 = rng.gen();
        let mut cumulative = 0.0;

        for pattern in &self.burst_patterns {
            cumulative += pattern.probability;
            if roll < cumulative {
                tracing::trace!(
                    game = %self.name,
                    pattern = %pattern.name,
                    "Burst pattern triggered"
                );
                return Some(pattern);
            }
        }

        None
    }

    pub fn generate_burst(&self, pattern: &BurstPattern, rng: &mut impl Rng) -> Vec<usize> {
        let count = rng.gen_range(pattern.packet_count_range.0..=pattern.packet_count_range.1);
        let mut packets = Vec::with_capacity(count);

        for _ in 0..count {
            let size = rng.gen_range(pattern.size_range.0..=pattern.size_range.1);
            packets.push(size);
        }

        tracing::trace!(
            game = %self.name,
            pattern = %pattern.name,
            burst_size = count,
            "Generated burst pattern"
        );

        packets
    }

    pub fn sample_entropy(&self, rng: &mut impl Rng) -> f64 {
        let normal =
            match Normal::new(self.entropy_profile.mean_entropy, self.entropy_profile.entropy_std) {
                Ok(n) => n,
                Err(_) => {
                    tracing::warn!(
                        game = %self.name,
                        "Failed to create entropy normal distribution"
                    );
                    return self.entropy_profile.mean_entropy;
                }
            };

        let raw = normal.sample(rng);
        raw.clamp(0.0, 8.0)
    }

    pub fn chi_squared_test(&self, samples: &[usize]) -> f64 {
        if samples.len() < 30 {
            tracing::warn!(
                sample_count = samples.len(),
                "Too few samples for chi-squared test"
            );
            return f64::INFINITY;
        }

        let num_bins = 10;
        let min_sample = *samples.iter().min().unwrap_or(&0);
        let max_sample = *samples.iter().max().unwrap_or(&0);

        if min_sample == max_sample {
            return 0.0;
        }

        let bin_width = ((max_sample - min_sample) as f64 / num_bins as f64).max(1.0);
        let mut observed = vec![0usize; num_bins];
        let mut expected = vec![0.0f64; num_bins];

        for &sample in samples {
            let bin = ((sample as f64 - min_sample as f64) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);
            observed[bin] += 1;
        }

        let total_samples = samples.len() as f64;
        for i in 0..num_bins {
            let bin_center = min_sample as f64 + (i as f64 + 0.5) * bin_width;
            let prob = self.size_probability_at(bin_center);
            expected[i] = prob * total_samples;
        }

        let mut chi_sq = 0.0;
        for i in 0..num_bins {
            if expected[i] > 0.0 {
                let diff = observed[i] as f64 - expected[i];
                chi_sq += (diff * diff) / expected[i];
            }
        }

        let dof = (num_bins - 1) as f64;
        let critical_value = Self::chi_squared_critical_value(dof, 0.05);

        tracing::info!(
            game = %self.name,
            chi_squared = chi_sq,
            critical_value = critical_value,
            dof = dof,
            sample_count = samples.len(),
            passes = chi_sq < critical_value,
            "Chi-squared goodness-of-fit test completed"
        );

        chi_sq
    }

    pub fn sample_target_size(&self, rng: &mut impl Rng) -> f64 {
        let u: f64 = rng.gen();
        let bucket = (u * (self.size_cdf.len() - 1) as f64) as usize;
        let bucket_next = (bucket + 1).min(self.size_cdf.len() - 1);
        let t = (u * (self.size_cdf.len() - 1) as f64) - bucket as f64;
        self.size_cdf[bucket] * (1.0 - t) + self.size_cdf[bucket_next] * t
    }

    pub fn sample_target_iat(&self, rng: &mut impl Rng) -> f64 {
        let u: f64 = rng.gen();
        let bucket = (u * (self.iat_cdf.len() - 1) as f64) as usize;
        let bucket_next = (bucket + 1).min(self.iat_cdf.len() - 1);
        let t = (u * (self.iat_cdf.len() - 1) as f64) - bucket as f64;
        self.iat_cdf[bucket] * (1.0 - t) + self.iat_cdf[bucket_next] * t
    }

    pub fn game_name(&self) -> &str {
        &self.name
    }

    pub fn client_server_ratio(&self) -> (f64, f64) {
        self.client_server_size_ratio
    }

    fn build_size_distribution(profile: &GamingProfile) -> PhantomResult<Vec<GaussianComponent>> {
        if profile.size_distribution.is_empty() {
            return Err(PhantomError::ConfigError(format!(
                "Profile '{}' has no size distribution components",
                profile.name
            )));
        }

        let total_weight: f64 = profile.size_distribution.iter().map(|c| c.weight).sum();
        if (total_weight - 1.0).abs() > 0.01 {
            tracing::warn!(
                game = %profile.name,
                total_weight = total_weight,
                "Size distribution weights do not sum to 1.0, normalizing"
            );
        }

        let components: Vec<GaussianComponent> = profile
            .size_distribution
            .iter()
            .map(|def| GaussianComponent {
                mean: def.mean,
                std_dev: def.std_dev.max(1.0),
                weight: if total_weight > 0.0 {
                    def.weight / total_weight
                } else {
                    1.0 / profile.size_distribution.len() as f64
                },
            })
            .collect();

        Ok(components)
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

    fn sample_from_cdf(&self, cdf: &[f64], rng: &mut impl Rng) -> usize {
        let roll: f64 = rng.gen();

        match cdf.binary_search_by(|&val| {
            if val < roll {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }) {
            Ok(idx) => idx,
            Err(idx) => idx.min(cdf.len() - 1),
        }
    }

    fn size_probability_at(&self, size: f64) -> f64 {
        let mut total_prob = 0.0;

        for component in &self.size_distribution {
            let normal = match Normal::new(component.mean, component.std_dev) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let pdf = normal.pdf(size);
            total_prob += pdf * component.weight;
        }

        total_prob
    }

    fn chi_squared_critical_value(dof: f64, alpha: f64) -> f64 {
        match dof.round() as usize {
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
            11 => 19.675,
            12 => 21.026,
            13 => 22.362,
            14 => 23.685,
            15 => 24.996,
            _ => {
                let z = match alpha {
                    0.05 => 1.645,
                    0.01 => 2.326,
                    _ => 1.96,
                };
                dof * (1.0 - 2.0 / (9.0 * dof) + z * (2.0 / (9.0 * dof)).sqrt()).powi(3)
            }
        }
    }

    fn compute_size_distribution_stats(components: &[GaussianComponent]) -> PacketSizeDistribution {
        let mean: f64 = components.iter().map(|c| c.mean * c.weight).sum();
        let variance: f64 = components
            .iter()
            .map(|c| c.weight * (c.std_dev.powi(2) + (c.mean - mean).powi(2)))
            .sum();
        let std = variance.sqrt();

        let mut rng = rand::thread_rng();
        let mut samples: Vec<f64> = Vec::with_capacity(8192);
        for _ in 0..8192 {
            let idx = Self::sample_from_cdf_static(
                &components.iter().map(|c| c.weight).collect::<Vec<_>>(),
                &mut rng,
            );
            let comp = &components[idx];
            if let Ok(normal) = Normal::new(comp.mean, comp.std_dev) {
                samples.push(normal.sample(&mut rng).clamp(40.0, 1500.0));
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = samples.len();
        let percentile = |p: f64| -> f64 {
            let idx = (p * (len - 1) as f64) as usize;
            samples[idx.min(len - 1)]
        };

        let skewness: f64 = samples
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / len as f64;
        let kurtosis: f64 = samples
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / len as f64;

        PacketSizeDistribution {
            mean,
            std,
            median: percentile(0.5),
            p25: percentile(0.25),
            p75: percentile(0.75),
            p95: percentile(0.95),
            min: samples.first().copied().unwrap_or(40.0),
            max: samples.last().copied().unwrap_or(1500.0),
            skewness,
            kurtosis,
        }
    }

    fn compute_timing_distribution_stats(
        bins: &[(f64, f64)],
    ) -> TimingDistribution {
        let mean_iat: f64 = bins.iter().map(|(t, p)| t * p).sum();
        let variance: f64 = bins
            .iter()
            .map(|(t, p)| p * (t - mean_iat).powi(2))
            .sum();
        let std_iat = variance.sqrt();
        let cv = if mean_iat > 0.0 {
            std_iat / mean_iat
        } else {
            0.0
        };

        let mut cum = 0.0;
        let mut p50 = mean_iat;
        let mut p95 = mean_iat;
        for &(t, p) in bins {
            cum += p;
            if cum >= 0.5 && p50 == mean_iat {
                p50 = t;
            }
            if cum >= 0.95 {
                p95 = t;
                break;
            }
        }

        TimingDistribution {
            mean_iat,
            std_iat,
            cv,
            p50_iat: p50,
            p95_iat: p95,
            burstiness: 0.3 + cv.min(1.0),
        }
    }

    fn build_empirical_cdf_from_mixture(
        components: &[GaussianComponent],
        bucket_count: usize,
    ) -> Vec<f64> {
        let weights: Vec<f64> = components.iter().map(|c| c.weight).collect();
        let cdf_weights = Self::compute_cdf(&weights);
        let mut rng = rand::thread_rng();
        let mut samples: Vec<f64> = Vec::with_capacity(4096);

        for _ in 0..4096 {
            let idx = Self::sample_from_cdf_static(&cdf_weights, &mut rng);
            let comp = &components[idx];
            if let Ok(normal) = Normal::new(comp.mean, comp.std_dev) {
                samples.push(normal.sample(&mut rng).clamp(40.0, 1500.0));
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut cdf = Vec::with_capacity(bucket_count);
        for i in 0..bucket_count {
            let idx = ((i as f64 / bucket_count as f64) * (samples.len() - 1) as f64) as usize;
            cdf.push(samples[idx.min(samples.len() - 1)]);
        }
        cdf
    }

    fn build_empirical_cdf_from_timing(
        bins: &[(f64, f64)],
        bucket_count: usize,
    ) -> Vec<f64> {
        let probs: Vec<f64> = bins.iter().map(|(_, p)| *p).collect();
        let cdf_probs = Self::compute_cdf(&probs);
        let mut rng = rand::thread_rng();
        let mut samples: Vec<f64> = Vec::with_capacity(4096);

        for _ in 0..4096 {
            let idx = Self::sample_from_cdf_static(&cdf_probs, &mut rng);
            let (base_time, _) = bins[idx];
            let jitter = rng.gen_range(-2.0..=2.0);
            samples.push((base_time + jitter).max(1.0));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut cdf = Vec::with_capacity(bucket_count);
        for i in 0..bucket_count {
            let idx = ((i as f64 / bucket_count as f64) * (samples.len() - 1) as f64) as usize;
            cdf.push(samples[idx.min(samples.len() - 1)]);
        }
        cdf
    }

    fn sample_from_cdf_static(cdf: &[f64], rng: &mut impl Rng) -> usize {
        let roll: f64 = rng.gen();
        match cdf.binary_search_by(|&val| {
            if val < roll {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }) {
            Ok(idx) => idx,
            Err(idx) => idx.min(cdf.len() - 1),
        }
    }
}

impl GaussianComponent {
    pub fn new(mean: f64, std_dev: f64, weight: f64) -> Self {
        Self {
            mean,
            std_dev: std_dev.max(1.0),
            weight,
        }
    }
}

impl BurstPattern {
    pub fn inter_packet_time(&self, rng: &mut impl Rng) -> f64 {
        rng.gen_range(self.inter_packet_time_range.0..=self.inter_packet_time_range.1)
    }

    pub fn packet_count(&self, rng: &mut impl Rng) -> usize {
        rng.gen_range(self.packet_count_range.0..=self.packet_count_range.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_profile() -> GamingProfile {
        GamingProfile {
            name: "test".to_string(),
            packet_size_min: 60,
            packet_size_max: 400,
            heartbeat_interval_ms: 30000,
            jitter_ms: 50,
            frequency_hz: 60,
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
        }
    }

    #[test]
    fn test_model_creation() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        assert_eq!(model.game_name(), "test");
        assert_eq!(model.size_distribution.len(), 3);
    }

    #[test]
    fn test_packet_size_distribution() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let sizes: Vec<usize> = (0..1000)
            .map(|_| model.sample_packet_size(&mut rng))
            .collect();
        let mean: f64 = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;

        assert!(
            mean > 80.0 && mean < 300.0,
            "Mean packet size {} out of expected range",
            mean
        );
    }

    #[test]
    fn test_timing_distribution() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let times: Vec<f64> = (0..1000)
            .map(|_| model.sample_inter_packet_time(&mut rng))
            .collect();
        let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;

        assert!(
            mean > 10.0 && mean < 40.0,
            "Mean inter-packet time {} out of expected range",
            mean
        );
    }

    #[test]
    fn test_chi_squared_test() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<usize> = (0..500)
            .map(|_| model.sample_packet_size(&mut rng))
            .collect();
        let chi_sq = model.chi_squared_test(&samples);

        assert!(chi_sq.is_finite(), "Chi-squared value should be finite");
    }

    #[test]
    fn test_entropy_sampling() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let entropies: Vec<f64> = (0..100)
            .map(|_| model.sample_entropy(&mut rng))
            .collect();
        let mean: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;

        assert!(
            mean > 6.0 && mean < 8.0,
            "Mean entropy {} out of expected range",
            mean
        );
        for &e in &entropies {
            assert!(e >= 0.0 && e <= 8.0, "Entropy {} out of valid range", e);
        }
    }

    #[test]
    fn test_burst_generation() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        if let Some(pattern) = model.should_burst(&mut rng) {
            let burst = model.generate_burst(pattern, &mut rng);
            assert!(!burst.is_empty());
            for &size in &burst {
                assert!(size >= pattern.size_range.0 && size <= pattern.size_range.1);
            }
        }
    }

    #[test]
    fn test_arc_rwlock_model() {
        let profile = test_profile();
        let model_arc = StatisticalGameTrafficModel::from_profile_arc(&profile).unwrap();
        let model = model_arc.read().unwrap();
        assert_eq!(model.game_name(), "test");
    }

    #[test]
    fn test_cdf_interpolation() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let size = model.sample_target_size(&mut rng);
            assert!(size >= model.size_dist.min);
            assert!(size <= model.size_dist.max);

            let iat = model.sample_target_iat(&mut rng);
            assert!(iat > 0.0);
        }
    }

    #[test]
    fn test_size_dist_stats_are_valid() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();

        assert!(model.size_dist.mean > 0.0);
        assert!(model.size_dist.std > 0.0);
        assert!(model.size_dist.p25 <= model.size_dist.median);
        assert!(model.size_dist.median <= model.size_dist.p75);
        assert!(model.size_dist.p75 <= model.size_dist.p95);
    }

    #[test]
    fn test_timing_dist_stats_are_valid() {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();

        assert!(model.timing_dist.mean_iat > 0.0);
        assert!(model.timing_dist.std_iat >= 0.0);
        assert!(model.timing_dist.p50_iat <= model.timing_dist.p95_iat);
    }
}
