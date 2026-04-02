use rand::Rng;
use rand_distr::{Distribution, Normal, Gamma, Exponential};
use std::collections::VecDeque;
use std::time::Duration;

use crate::masking::profiles::{GamingProfile, BurstPatternDef, GaussianComponentDef, EntropyProfileDef};
use crate::utils::{PhantomError, PhantomResult};

// ─── Burst-Specific Traffic Model ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StatisticalGameTrafficModel {
    pub game_name: String,
    pub size_components: Vec<GaussianComponent>,
    pub timing_bins: Vec<TimingBin>,
    pub burst_patterns: Vec<BurstPatternModel>,
    pub client_ratio: f64,
    pub server_ratio: f64,
    pub mean_entropy: f64,
    pub entropy_std: f64,
    pub cooldown_duration_mean_ms: f64,
    pub cooldown_duration_std_ms: f64,
    pub inter_burst_interval_mean_s: f64,
    pub inter_burst_interval_std_s: f64,
}

#[derive(Debug, Clone)]
pub struct GaussianComponent {
    pub mean: f64,
    pub std_dev: f64,
    pub weight: f64,
    pub normal_dist: Normal<f64>,
}

#[derive(Debug, Clone)]
pub struct TimingBin {
    pub time_ms: f64,
    pub probability: f64,
    pub cumulative_prob: f64,
}

#[derive(Debug, Clone)]
pub struct BurstPatternModel {
    pub name: String,
    pub burst_type: BurstType,
    pub packet_count_dist: Gamma<f64>,
    pub packet_count_min: usize,
    pub packet_count_max: usize,
    pub inter_packet_delay_dist: Gamma<f64>,
    pub inter_packet_delay_min_ms: f64,
    pub inter_packet_delay_max_ms: f64,
    pub size_dist: Normal<f64>,
    pub size_min: usize,
    pub size_max: usize,
    pub probability: f64,
    pub cumulative_prob: f64,
}

impl StatisticalGameTrafficModel {
    pub fn from_profile(profile: &GamingProfile) -> PhantomResult<Self> {
        let size_components = profile
            .size_distribution
            .iter()
            .map(|def| {
                let normal = Normal::new(def.mean, def.std_dev.abs()).map_err(|e| {
                    PhantomError::ConfigError(format!("Invalid normal distribution params: {}", e))
                })?;
                Ok(GaussianComponent {
                    mean: def.mean,
                    std_dev: def.std_dev,
                    weight: def.weight,
                    normal_dist,
                })
            })
            .collect::<PhantomResult<Vec<_>>>()?;

        let mut cumulative = 0.0;
        let timing_bins: Vec<TimingBin> = profile
            .timing_distribution
            .iter()
            .map(|def| {
                cumulative += def.probability;
                TimingBin {
                    time_ms: def.time_ms,
                    probability: def.probability,
                    cumulative_prob: cumulative,
                }
            })
            .collect();

        let total_prob = timing_bins.last().map(|b| b.cumulative_prob).unwrap_or(0.0);
        if (total_prob - 1.0).abs() > 0.01 {
            tracing::warn!(
                total_prob = total_prob,
                "Timing distribution probabilities do not sum to 1.0, normalizing"
            );
        }

        let burst_patterns = Self::build_burst_pattern_models(&profile.burst_patterns)?;

        let (mean_entropy, entropy_std) = profile
            .entropy_profile
            .as_ref()
            .map(|ep| (ep.mean_entropy, ep.entropy_std))
            .unwrap_or((7.0, 0.3));

        let cooldown_mean = 2500.0;
        let cooldown_std = 800.0;
        let inter_burst_mean = 8.0;
        let inter_burst_std = 4.0;

        tracing::info!(
            game = %profile.name,
            size_components = size_components.len(),
            burst_patterns = burst_patterns.len(),
            "Built StatisticalGameTrafficModel"
        );

        Ok(Self {
            game_name: profile.name.clone(),
            size_components,
            timing_bins,
            burst_patterns,
            client_ratio: profile.client_server_size_ratio.0,
            server_ratio: profile.client_server_size_ratio.1,
            mean_entropy,
            entropy_std,
            cooldown_duration_mean_ms: cooldown_mean,
            cooldown_duration_std_ms: cooldown_std,
            inter_burst_interval_mean_s: inter_burst_mean,
            inter_burst_interval_std_s: inter_burst_std,
        })
    }

    fn build_burst_pattern_models(
        patterns: &[BurstPatternDef],
    ) -> PhantomResult<Vec<BurstPatternModel>> {
        let mut models = Vec::with_capacity(patterns.len());
        let mut cumulative_prob = 0.0;

        for def in patterns {
            let burst_type = Self::classify_burst_type(&def.name);

            let count_mean = (def.packet_count_range.0 + def.packet_count_range.1) as f64 / 2.0;
            let count_range = (def.packet_count_range.1 - def.packet_count_range.0) as f64;
            let count_std = (count_range / 4.0).max(0.5);
            let count_shape = (count_mean / count_std).powi(2);
            let count_rate = count_mean / (count_std * count_std);
            let count_dist = Gamma::new(count_shape, count_rate).map_err(|e| {
                PhantomError::ConfigError(format!("Invalid gamma distribution for packet count: {}", e))
            })?;

            let ipd_mean = (def.inter_packet_time_range.0 + def.inter_packet_time_range.1) / 2.0;
            let ipd_range = def.inter_packet_time_range.1 - def.inter_packet_time_range.0;
            let ipd_std = (ipd_range / 4.0).max(0.5);
            let ipd_shape = (ipd_mean / ipd_std).powi(2);
            let ipd_rate = ipd_mean / (ipd_std * ipd_std);
            let ipd_dist = Gamma::new(ipd_shape, ipd_rate).map_err(|e| {
                PhantomError::ConfigError(format!("Invalid gamma distribution for inter-packet delay: {}", e))
            })?;

            let size_mean = (def.size_range.0 + def.size_range.1) as f64 / 2.0;
            let size_range = (def.size_range.1 - def.size_range.0) as f64;
            let size_std = (size_range / 4.0).max(1.0);
            let size_dist = Normal::new(size_mean, size_std).map_err(|e| {
                PhantomError::ConfigError(format!("Invalid normal distribution for packet size: {}", e))
            })?;

            cumulative_prob += def.probability;

            models.push(BurstPatternModel {
                name: def.name.clone(),
                burst_type,
                packet_count_dist: count_dist,
                packet_count_min: def.packet_count_range.0,
                packet_count_max: def.packet_count_range.1,
                inter_packet_delay_dist: ipd_dist,
                inter_packet_delay_min_ms: def.inter_packet_time_range.0,
                inter_packet_delay_max_ms: def.inter_packet_time_range.1,
                size_dist,
                size_min: def.size_range.0,
                size_max: def.size_range.1,
                probability: def.probability,
                cumulative_prob,
            });
        }

        if cumulative_prob > 0.0 && (cumulative_prob - 1.0).abs() > 0.01 {
            tracing::warn!(
                cumulative_prob = cumulative_prob,
                "Burst pattern probabilities do not sum to 1.0, normalizing"
            );
            for model in &mut models {
                model.cumulative_prob /= cumulative_prob;
                model.probability /= cumulative_prob;
            }
        }

        Ok(models)
    }

    fn classify_burst_type(name: &str) -> BurstType {
        match name.to_lowercase().as_str() {
            n if n.contains("shoot") || n.contains("fire") || n.contains("attack") => BurstType::Shooting,
            n if n.contains("move") || n.contains("position") => BurstType::Movement,
            n if n.contains("map") || n.contains("load") || n.contains("chunk") => BurstType::MapLoad,
            n if n.contains("voice") || n.contains("chat") || n.contains("audio") => BurstType::VoiceChat,
            n if n.contains("inventory") || n.contains("menu") || n.contains("item") => BurstType::Inventory,
            _ => BurstType::Movement,
        }
    }
}

// ─── Burst State ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum BurstState {
    Idle {
        last_activity: std::time::Instant,
        time_since_last_burst: Duration,
    },
    BurstActive {
        packets_remaining: usize,
        inter_packet_delay: Duration,
        burst_type: BurstType,
        current_packet_index: usize,
        total_packets: usize,
    },
    PostBurstCooldown {
        cooldown_remaining: Duration,
        burst_type_just_finished: BurstType,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurstType {
    Shooting,
    Movement,
    MapLoad,
    VoiceChat,
    Inventory,
}

impl std::fmt::Display for BurstType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BurstType::Shooting => write!(f, "shooting"),
            BurstType::Movement => write!(f, "movement"),
            BurstType::MapLoad => write!(f, "map_load"),
            BurstType::VoiceChat => write!(f, "voice_chat"),
            BurstType::Inventory => write!(f, "inventory"),
        }
    }
}

// ─── Packet Actions ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum PacketAction {
    Send { data: Vec<u8>, delay: Duration },
    Pad { size: usize, delay: Duration },
    Split { data: Vec<Vec<u8>>, delay: Duration },
    Keepalive { delay: Duration },
}

// ─── Burst Engine ─────────────────────────────────────────────────────────────

pub struct BurstEngine {
    game_model: StatisticalGameTrafficModel,
    state: BurstState,
    pending_data: VecDeque<Vec<u8>>,
    sequence_counter: u32,
    rng: rand::rngs::ThreadRng,
    total_bursts_generated: u64,
    total_packets_sent: u64,
    last_inter_burst_check: std::time::Instant,
}

impl BurstEngine {
    pub fn new(game_model: StatisticalGameTrafficModel) -> Self {
        tracing::info!(
            game = %game_model.game_name,
            "Initialized BurstEngine"
        );
        Self {
            state: BurstState::Idle {
                last_activity: std::time::Instant::now(),
                time_since_last_burst: Duration::from_secs(0),
            },
            game_model,
            pending_data: VecDeque::new(),
            sequence_counter: 0,
            rng: rand::thread_rng(),
            total_bursts_generated: 0,
            total_packets_sent: 0,
            last_inter_burst_check: std::time::Instant::now(),
        }
    }

    pub fn process(&mut self, data: Vec<u8>) -> Vec<PacketAction> {
        if !data.is_empty() {
            self.pending_data.push_back(data);
        }

        self.check_burst_transition();

        let actions = match &self.state {
            BurstState::Idle { .. } => self.handle_idle(),
            BurstState::BurstActive { .. } => self.generate_burst_packets(),
            BurstState::PostBurstCooldown { .. } => self.handle_cooldown(),
        };

        self.total_packets_sent += actions.len() as u64;
        actions
    }

    fn check_burst_transition(&mut self) {
        let now = std::time::Instant::now();

        match &self.state {
            BurstState::Idle { .. } => {
                let elapsed_since_check = self.last_inter_burst_check.elapsed();
                let inter_burst_target = self.sample_inter_burst_interval();

                if elapsed_since_check >= inter_burst_target || !self.pending_data.is_empty() {
                    if let Some(pattern) = self.select_burst_pattern() {
                        let packet_count = self.sample_packet_count(pattern);
                        let ipd = self.sample_inter_packet_delay(pattern);

                        tracing::debug!(
                            burst_type = %pattern.burst_type,
                            packet_count = packet_count,
                            inter_packet_delay_ms = ipd.as_millis(),
                            "Transitioning to burst state"
                        );

                        self.state = BurstState::BurstActive {
                            packets_remaining: packet_count,
                            inter_packet_delay: ipd,
                            burst_type: pattern.burst_type,
                            current_packet_index: 0,
                            total_packets: packet_count,
                        };

                        self.total_bursts_generated += 1;
                        self.last_inter_burst_check = now;
                    }
                }
            }
            BurstState::BurstActive {
                packets_remaining,
                burst_type,
                ..
            } => {
                if *packets_remaining == 0 {
                    let cooldown = self.sample_cooldown_duration(burst_type);
                    tracing::debug!(
                        burst_type = %burst_type,
                        cooldown_ms = cooldown.as_millis(),
                        "Burst complete, entering cooldown"
                    );
                    self.state = BurstState::PostBurstCooldown {
                        cooldown_remaining: cooldown,
                        burst_type_just_finished: *burst_type,
                    };
                }
            }
            BurstState::PostBurstCooldown {
                cooldown_remaining, ..
            } => {
                if *cooldown_remaining == Duration::from_secs(0) {
                    tracing::debug!("Cooldown complete, returning to idle");
                    self.state = BurstState::Idle {
                        last_activity: now,
                        time_since_last_burst: Duration::from_secs(0),
                    };
                }
            }
        }
    }

    fn generate_burst_packets(&mut self) -> Vec<PacketAction> {
        let (packets_remaining, inter_packet_delay, burst_type, current_index, total) = match &self.state {
            BurstState::BurstActive {
                packets_remaining,
                inter_packet_delay,
                burst_type,
                current_packet_index,
                total_packets,
            } => (
                *packets_remaining,
                *inter_packet_delay,
                *burst_type,
                *current_packet_index,
                *total_packets,
            ),
            _ => return Vec::new(),
        };

        if packets_remaining == 0 {
            return Vec::new();
        }

        let mut actions = Vec::new();
        let pattern = self.find_pattern_for_type(burst_type);

        let packets_this_tick = match burst_type {
            BurstType::Shooting => self.rng.gen_range(1..=3),
            BurstType::Movement => self.rng.gen_range(1..=2),
            BurstType::MapLoad => self.rng.gen_range(2..=5),
            BurstType::VoiceChat => 1,
            BurstType::Inventory => self.rng.gen_range(1..=2),
        };

        let packets_to_send = packets_this_tick.min(packets_remaining);

        for i in 0..packets_to_send {
            let packet_size = self.sample_packet_size(pattern);
            let delay = if i == 0 {
                inter_packet_delay
            } else {
                inter_packet_delay.mul_f64(0.6)
            };

            if let Some(data) = self.pending_data.pop_front() {
                if data.len() > packet_size {
                    let chunks: Vec<Vec<u8>> = data
                        .chunks(packet_size)
                        .map(|chunk| chunk.to_vec())
                        .collect();
                    actions.push(PacketAction::Split {
                        data: chunks,
                        delay,
                    });
                } else {
                    actions.push(PacketAction::Send { data, delay });
                    if data.len() < packet_size {
                        actions.push(PacketAction::Pad {
                            size: packet_size - data.len(),
                            delay: Duration::from_micros(100),
                        });
                    }
                }
            } else {
                actions.push(self.generate_filler_packet(packet_size, delay));
            }
        }

        let new_remaining = packets_remaining.saturating_sub(packets_to_send);
        self.state = BurstState::BurstActive {
            packets_remaining: new_remaining,
            inter_packet_delay,
            burst_type,
            current_packet_index: current_index + packets_to_send,
            total_packets: total,
        };

        tracing::trace!(
            burst_type = %burst_type,
            packets_sent = packets_to_send,
            remaining = new_remaining,
            "Generated burst packets"
        );

        actions
    }

    fn handle_idle(&mut self) -> Vec<PacketAction> {
        let mut actions = Vec::new();

        if self.pending_data.is_empty() {
            let keepalive_delay = self.sample_idle_keepalive_delay();
            actions.push(PacketAction::Keepalive {
                delay: keepalive_delay,
            });
        } else {
            let to_release = self.rng.gen_range(1..=2).min(self.pending_data.len());
            for _ in 0..to_release {
                if let Some(data) = self.pending_data.pop_front() {
                    let size = self.sample_idle_packet_size();
                    let delay = self.sample_idle_keepalive_delay();

                    if data.len() > size {
                        let chunk = data[..size].to_vec();
                        let remainder = data[size..].to_vec();
                        self.pending_data.push_front(remainder);
                        actions.push(PacketAction::Send {
                            data: chunk,
                            delay,
                        });
                    } else {
                        actions.push(PacketAction::Send { data, delay });
                        if data.len() < size {
                            actions.push(PacketAction::Pad {
                                size: size - data.len(),
                                delay: Duration::from_micros(50),
                            });
                        }
                    }
                }
            }

            tracing::trace!(
                packets_released = to_release,
                pending_remaining = self.pending_data.len(),
                "Released packets during idle state"
            );
        }

        actions
    }

    fn handle_cooldown(&mut self) -> Vec<PacketAction> {
        let mut actions = Vec::new();

        let (cooldown_remaining, burst_type) = match &self.state {
            BurstState::PostBurstCooldown {
                cooldown_remaining,
                burst_type_just_finished,
            } => (*cooldown_remaining, *burst_type_just_finished),
            _ => return actions,
        };

        let cooldown_progress = 1.0 - (cooldown_remaining.as_millis() as f64
            / self.game_model.cooldown_duration_mean_ms);

        let activity_level = cooldown_progress.powi(2);

        if self.rng.gen_bool(activity_level.min(0.8)) {
            let size = self.sample_cooldown_packet_size(burst_type);
            let delay = self.sample_cooldown_delay();

            if let Some(data) = self.pending_data.pop_front() {
                actions.push(PacketAction::Send {
                    data: if data.len() > size {
                        let chunk = data[..size].to_vec();
                        self.pending_data.push_front(data[size..].to_vec());
                        chunk
                    } else {
                        data
                    },
                    delay,
                });
            } else {
                actions.push(self.generate_filler_packet(size, delay));
            }
        }

        let elapsed = self.last_inter_burst_check.elapsed();
        let new_remaining = cooldown_remaining.saturating_sub(elapsed);
        self.state = BurstState::PostBurstCooldown {
            cooldown_remaining: new_remaining,
            burst_type_just_finished: burst_type,
        };

        tracing::trace!(
            cooldown_remaining_ms = new_remaining.as_millis(),
            activity_level = activity_level,
            "Cooldown state update"
        );

        actions
    }

    pub fn next_action_delay(&self) -> Duration {
        match &self.state {
            BurstState::Idle { .. } => self.sample_idle_keepalive_delay(),
            BurstState::BurstActive {
                inter_packet_delay, ..
            } => *inter_packet_delay,
            BurstState::PostBurstCooldown {
                cooldown_remaining, ..
            } => {
                Duration::from_millis(200).min(*cooldown_remaining)
            }
        }
    }

    fn sample_packet_count(&mut self, pattern: &BurstPatternModel) -> usize {
        let raw = pattern.packet_count_dist.sample(&mut self.rng);
        raw.clamp(pattern.packet_count_min as f64, pattern.packet_count_max as f64) as usize
    }

    fn sample_inter_packet_delay(&mut self, pattern: &BurstPatternModel) -> Duration {
        let raw = pattern.inter_packet_delay_dist.sample(&mut self.rng);
        let clamped = raw.clamp(
            pattern.inter_packet_delay_min_ms,
            pattern.inter_packet_delay_max_ms,
        );
        Duration::from_millis(clamped as u64)
    }

    fn sample_packet_size(&mut self, pattern: &BurstPatternModel) -> usize {
        let raw = pattern.size_dist.sample(&mut self.rng);
        raw.clamp(pattern.size_min as f64, pattern.size_max as f64) as usize
    }

    fn sample_idle_packet_size(&mut self) -> usize {
        let weight_sum: f64 = self.game_model.size_components.iter().map(|c| c.weight).sum();
        let mut r = self.rng.gen_range(0.0..weight_sum);

        for component in &self.game_model.size_components {
            r -= component.weight;
            if r <= 0.0 {
                let size = component.normal_dist.sample(&mut self.rng);
                let (min, max) = self.game_model
                    .size_components
                    .iter()
                    .map(|c| {
                        let lo = c.mean - 3.0 * c.std_dev;
                        let hi = c.mean + 3.0 * c.std_dev;
                        (lo, hi)
                    })
                    .fold((f64::MAX, f64::MIN), |(a_lo, a_hi), (b_lo, b_hi)| {
                        (a_lo.min(b_lo), a_hi.max(b_hi))
                    });
                return size.clamp(min.max(32.0), max.min(1500.0)) as usize;
            }
        }

        self.game_model.size_components[0].mean as usize
    }

    fn sample_idle_keepalive_delay(&mut self) -> Duration {
        let rate = 1.0 / self.game_model.inter_burst_interval_mean_s;
        let exponential_dist = Exponential::new(rate).unwrap_or_else(|_| Exponential::new(0.125).unwrap());
        let seconds = exponential_dist.sample(&mut self.rng);
        Duration::from_millis((seconds * 1000.0).clamp(50.0, 5000.0) as u64)
    }

    fn sample_cooldown_duration(&mut self, burst_type: &BurstType) -> Duration {
        let (mean_mult, std_mult) = match burst_type {
            BurstType::Shooting => (1.2, 1.0),
            BurstType::Movement => (0.6, 0.8),
            BurstType::MapLoad => (1.5, 1.2),
            BurstType::VoiceChat => (0.4, 0.5),
            BurstType::Inventory => (0.8, 0.6),
        };

        let mean = self.game_model.cooldown_duration_mean_ms * mean_mult;
        let std = self.game_model.cooldown_duration_std_ms * std_mult;
        let normal = Normal::new(mean, std).unwrap_or_else(|_| Normal::new(2500.0, 800.0).unwrap());
        let ms = normal.sample(&mut self.rng).clamp(500.0, 8000.0);
        Duration::from_millis(ms as u64)
    }

    fn sample_cooldown_packet_size(&mut self, burst_type: BurstType) -> usize {
        let base_size = self.sample_idle_packet_size();
        let reduction = match burst_type {
            BurstType::Shooting => 0.7,
            BurstType::Movement => 0.8,
            BurstType::MapLoad => 0.5,
            BurstType::VoiceChat => 0.9,
            BurstType::Inventory => 0.6,
        };
        (base_size as f64 * reduction).clamp(32.0, 500.0) as usize
    }

    fn sample_cooldown_delay(&mut self) -> Duration {
        let rate = 0.5;
        let exponential = Exponential::new(rate).unwrap_or_else(|_| Exponential::new(0.5).unwrap());
        let seconds = exponential.sample(&mut self.rng);
        Duration::from_millis((seconds * 1000.0).clamp(200.0, 3000.0) as u64)
    }

    fn sample_inter_burst_interval(&mut self) -> Duration {
        let mean = self.game_model.inter_burst_interval_mean_s;
        let std = self.game_model.inter_burst_interval_std_s;
        let shape = (mean / std).powi(2);
        let rate = mean / (std * std);
        let gamma = Gamma::new(shape, rate).unwrap_or_else(|_| Gamma::new(4.0, 0.5).unwrap());
        let seconds = gamma.sample(&mut self.rng);
        Duration::from_millis((seconds * 1000.0).clamp(1000.0, 30000.0) as u64)
    }

    fn select_burst_pattern(&mut self) -> Option<&BurstPatternModel> {
        if self.game_model.burst_patterns.is_empty() {
            tracing::warn!("No burst patterns available in model");
            return None;
        }

        let r = self.rng.gen_range(0.0..1.0);

        for pattern in &self.game_model.burst_patterns {
            if r <= pattern.cumulative_prob {
                return Some(pattern);
            }
        }

        self.game_model.burst_patterns.last()
    }

    fn find_pattern_for_type(&self, burst_type: BurstType) -> &BurstPatternModel {
        self.game_model
            .burst_patterns
            .iter()
            .find(|p| p.burst_type == burst_type)
            .unwrap_or_else(|| {
                tracing::warn!(
                    burst_type = %burst_type,
                    "No pattern found for burst type, using first available"
                );
                &self.game_model.burst_patterns[0]
            })
    }

    fn generate_filler_packet(&mut self, size: usize, delay: Duration) -> PacketAction {
        let mut data = vec![0u8; size];
        self.rng.fill(&mut data[..]);

        if size >= 24 {
            let conv: u32 = self.rng.gen_range(1..=u32::MAX);
            let cmd: u8 = 0x81;
            let frg: u8 = 0;
            let wnd: u16 = 32;
            let ts: u32 = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_millis() & 0xFFFFFFFF) as u32;
            let sn: u32 = self.sequence_counter;
            let una: u32 = self.sequence_counter.saturating_sub(1);
            let len: u32 = size as u32;

            data[0..4].copy_from_slice(&conv.to_le_bytes());
            data[4] = cmd;
            data[5] = frg;
            data[6..8].copy_from_slice(&wnd.to_le_bytes());
            data[8..12].copy_from_slice(&ts.to_le_bytes());
            data[12..16].copy_from_slice(&sn.to_le_bytes());
            data[16..20].copy_from_slice(&una.to_le_bytes());
            data[20..24].copy_from_slice(&len.to_le_bytes());

            self.sequence_counter = self.sequence_counter.wrapping_add(1);
        }

        tracing::trace!(
            size = size,
            delay_ms = delay.as_millis(),
            "Generated filler packet"
        );

        PacketAction::Send { data, delay }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> StatisticalGameTrafficModel {
        let profile = GamingProfile {
            name: "test_game".to_string(),
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
                crate::masking::profiles::TimingBinDef { time_ms: 16.67, probability: 0.70 },
                crate::masking::profiles::TimingBinDef { time_ms: 33.33, probability: 0.15 },
                crate::masking::profiles::TimingBinDef { time_ms: 50.0, probability: 0.10 },
                crate::masking::profiles::TimingBinDef { time_ms: 100.0, probability: 0.05 },
            ],
            burst_patterns: vec![
                BurstPatternDef {
                    name: "shooting".to_string(),
                    packet_count_range: (5, 10),
                    inter_packet_time_range: (5.0, 15.0),
                    size_range: (60, 150),
                    probability: 0.4,
                },
                BurstPatternDef {
                    name: "movement".to_string(),
                    packet_count_range: (2, 4),
                    inter_packet_time_range: (15.0, 20.0),
                    size_range: (60, 100),
                    probability: 0.6,
                },
            ],
            client_server_size_ratio: (0.6, 1.0),
            entropy_profile: Some(EntropyProfileDef { mean_entropy: 7.2, entropy_std: 0.3 }),
        };
        StatisticalGameTrafficModel::from_profile(&profile).unwrap()
    }

    #[test]
    fn test_burst_engine_creation() {
        let model = make_test_model();
        let engine = BurstEngine::new(model);
        assert!(matches!(engine.state, BurstState::Idle { .. }));
        assert_eq!(engine.pending_data.len(), 0);
    }

    #[test]
    fn test_process_with_data_generates_actions() {
        let model = make_test_model();
        let mut engine = BurstEngine::new(model);
        let actions = engine.process(vec![1, 2, 3, 4, 5]);
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_process_empty_data_still_generates_keepalive() {
        let model = make_test_model();
        let mut engine = BurstEngine::new(model);
        let actions = engine.process(Vec::new());
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_burst_type_classification() {
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("shooting"), BurstType::Shooting);
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("fire_rate"), BurstType::Shooting);
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("movement"), BurstType::Movement);
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("map_load"), BurstType::MapLoad);
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("voice_chat"), BurstType::VoiceChat);
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("inventory"), BurstType::Inventory);
        assert_eq!(StatisticalGameTrafficModel::classify_burst_type("unknown"), BurstType::Movement);
    }

    #[test]
    fn test_packet_action_variants() {
        let model = make_test_model();
        let mut engine = BurstEngine::new(model);

        for i in 0..50 {
            let _ = engine.process(vec![i as u8; 100]);
        }
    }

    #[test]
    fn test_next_action_delay_returns_valid_duration() {
        let model = make_test_model();
        let engine = BurstEngine::new(model);
        let delay = engine.next_action_delay();
        assert!(delay > Duration::from_secs(0));
        assert!(delay < Duration::from_secs(10));
    }
}
