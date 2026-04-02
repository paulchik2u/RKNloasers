use rand::Rng;
use std::time::{Duration, Instant};

use crate::masking::profiles::GamingProfile;

pub struct TrafficShaper {
    profile: GamingProfile,
    rng: rand::rngs::ThreadRng,
    last_heartbeat: Instant,
}

impl TrafficShaper {
    pub fn new(profile: GamingProfile) -> Self {
        tracing::info!(
            profile = %profile.name,
            heartbeat_ms = profile.heartbeat_interval_ms,
            jitter_ms = profile.jitter_ms,
            frequency_hz = profile.frequency_hz,
            "Initialized TrafficShaper"
        );
        TrafficShaper {
            profile,
            rng: rand::thread_rng(),
            last_heartbeat: Instant::now(),
        }
    }

    pub fn should_send_heartbeat(&self) -> bool {
        let elapsed = self.last_heartbeat.elapsed();
        let interval = Duration::from_millis(self.profile.heartbeat_interval_ms);
        elapsed >= interval
    }

    pub fn get_jittered_delay(&self) -> Duration {
        let jitter_range = self.profile.jitter_ms as f64;
        let jitter = self.rng.gen_range(-jitter_range..=jitter_range);
        let jitter_ms = jitter.max(0.0) as u64;
        Duration::from_millis(jitter_ms)
    }

    pub fn get_send_interval(&self) -> Duration {
        let base_interval_ms = 1000u64.saturating_div(self.profile.frequency_hz as u64);
        let jitter = self.get_jittered_delay();
        let total_ms = base_interval_ms.saturating_add(jitter.as_millis() as u64);
        Duration::from_millis(total_ms)
    }

    pub fn heartbeat_packet(&self) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let conv: u32 = rng.gen_range(1..=u32::MAX);
        let cmd: u8 = 0x81;
        let frg: u8 = 0;
        let wnd: u16 = 32;
        let ts: u32 = rng.gen();
        let sn: u32 = rng.gen();
        let una: u32 = 0;

        let heartbeat_payload_size = self.rng.gen_range(8..=32);
        let len: u32 = (24 + heartbeat_payload_size) as u32;

        let mut packet = Vec::with_capacity(len as usize);
        packet.extend_from_slice(&conv.to_le_bytes());
        packet.push(cmd);
        packet.push(frg);
        packet.extend_from_slice(&wnd.to_le_bytes());
        packet.extend_from_slice(&ts.to_le_bytes());
        packet.extend_from_slice(&sn.to_le_bytes());
        packet.extend_from_slice(&una.to_le_bytes());
        packet.extend_from_slice(&len.to_le_bytes());

        let mut payload: Vec<u8> = vec![0u8; heartbeat_payload_size as usize];
        rng.fill(&mut payload[..]);
        packet.extend_from_slice(&payload);

        tracing::trace!(
            packet_size = packet.len(),
            "Generated heartbeat packet"
        );
        packet
    }

    pub fn mark_heartbeat_sent(&mut self) {
        self.last_heartbeat = Instant::now();
        tracing::trace!("Heartbeat timestamp updated");
    }

    pub fn profile(&self) -> &GamingProfile {
        &self.profile
    }
}
