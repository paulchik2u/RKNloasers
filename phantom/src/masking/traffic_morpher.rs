use rand::Rng;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::masking::adversarial::{AdversarialMasker, MaskingDecision};
use crate::masking::burst_engine::BurstEngine;
use crate::masking::statistical_model::StatisticalGameTrafficModel;
use crate::utils::{PhantomError, PhantomResult};

// ─── Constants ────────────────────────────────────────────────────────────────

const HEADER_SIZE: usize = 24;
const MIN_PAYLOAD_SIZE: usize = 16;
const MAX_GAME_PACKET_SIZE: usize = 1400;
const MAGIC_BYTE: u8 = 0xA7;

// ─── Public Types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MorphedPacket {
    pub data: Vec<u8>,
    pub delay: Duration,
    pub packet_type: GamePacketType,
    pub sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePacketType {
    Movement,
    Action,
    State,
    Keepalive,
    Voice,
}

impl GamePacketType {
    fn to_byte(self) -> u8 {
        match self {
            GamePacketType::Movement => 0x01,
            GamePacketType::Action => 0x02,
            GamePacketType::State => 0x03,
            GamePacketType::Keepalive => 0x04,
            GamePacketType::Voice => 0x05,
        }
    }

    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(GamePacketType::Movement),
            0x02 => Some(GamePacketType::Action),
            0x03 => Some(GamePacketType::State),
            0x04 => Some(GamePacketType::Keepalive),
            0x05 => Some(GamePacketType::Voice),
            _ => None,
        }
    }

    fn select_by_size(size: usize, rng: &mut impl Rng) -> Self {
        if size < 80 {
            let r: u8 = rng.gen_range(0..4);
            match r {
                0 => GamePacketType::Movement,
                1 => GamePacketType::Keepalive,
                2 => GamePacketType::State,
                _ => GamePacketType::Movement,
            }
        } else if size < 300 {
            let r: u8 = rng.gen_range(0..3);
            match r {
                0 => GamePacketType::Action,
                1 => GamePacketType::State,
                _ => GamePacketType::Movement,
            }
        } else if size < 600 {
            let r: u8 = rng.gen_range(0..3);
            match r {
                0 => GamePacketType::Voice,
                1 => GamePacketType::Action,
                _ => GamePacketType::State,
            }
        } else {
            GamePacketType::Action
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FragmentState {
    current_fragment: usize,
    total_fragments: usize,
    fragment_size: usize,
}

impl FragmentState {
    fn new(current: usize, total: usize, frag_size: usize) -> Self {
        Self {
            current_fragment: current,
            total_fragments: total,
            fragment_size: frag_size,
        }
    }

    fn encode(&self) -> u8 {
        let total_clamped = (self.total_fragments as u8).min(0x0F);
        let current_clamped = (self.current_fragment as u8).min(0x0F);
        (current_clamped << 4) | total_clamped
    }

    fn decode(byte: u8) -> Self {
        let total = (byte & 0x0F) as usize;
        let current = ((byte >> 4) & 0x0F) as usize;
        Self {
            current_fragment: current,
            total_fragments: total,
            fragment_size: 0,
        }
    }
}

// ─── Reassembly Buffer ────────────────────────────────────────────────────────

#[derive(Debug)]
struct ReassemblyBuffer {
    fragments: BTreeMap<usize, Vec<u8>>,
    total_fragments: usize,
    original_length: usize,
}

impl ReassemblyBuffer {
    fn new(total_fragments: usize, original_length: usize) -> Self {
        Self {
            fragments: BTreeMap::new(),
            total_fragments,
            original_length,
        }
    }

    fn insert(&mut self, index: usize, data: Vec<u8>) {
        self.fragments.insert(index, data);
    }

    fn is_complete(&self) -> bool {
        self.fragments.len() == self.total_fragments
            && self.fragments.keys().copied().eq(0..self.total_fragments)
    }

    fn reassemble(self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.original_length);
        for (_, fragment) in self.fragments {
            output.extend_from_slice(&fragment);
        }
        output.truncate(self.original_length);
        output
    }
}

// ─── Traffic Morpher ──────────────────────────────────────────────────────────

pub struct TrafficMorpher {
    statistical_model: StatisticalGameTrafficModel,
    burst_engine: BurstEngine,
    adversarial_masker: AdversarialMasker,
    data_buffer: Vec<u8>,
    sequence: u64,
    fragment_state: FragmentState,
    reassembly_buffers: BTreeMap<u64, ReassemblyBuffer>,
    conversation_id: u32,
    rng: rand::rngs::ThreadRng,
}

impl TrafficMorpher {
    pub fn new(game_model: StatisticalGameTrafficModel) -> PhantomResult<Self> {
        let model_clone = game_model.clone();
        let burst_engine = BurstEngine::new(game_model.clone());
        let adversarial_masker = AdversarialMasker::new(game_model)?;

        let mut rng = rand::thread_rng();
        let conversation_id: u32 = rng.gen();

        tracing::info!(
            game = %game_model.name,
            conversation_id = conversation_id,
            "Initialized TrafficMorpher"
        );

        Ok(Self {
            statistical_model: game_model,
            burst_engine,
            adversarial_masker,
            data_buffer: Vec::new(),
            sequence: 0,
            fragment_state: FragmentState::default(),
            reassembly_buffers: BTreeMap::new(),
            conversation_id,
            rng,
        })
    }

    // ─── Main Morphing Interface ──────────────────────────────────────────

    pub fn morph(&mut self, data: Vec<u8>) -> Vec<MorphedPacket> {
        if data.is_empty() {
            return self.generate_keepalive_packets();
        }

        tracing::trace!(
            input_size = data.len(),
            sequence = self.sequence,
            "Morphing traffic data"
        );

        let fragments = self.fragment_data(data);
        let total_fragments = fragments.len();

        let mut morphed_packets = Vec::with_capacity(total_fragments);

        for (idx, fragment) in fragments.into_iter().enumerate() {
            let frag_state = FragmentState::new(idx, total_fragments, fragment.len());
            let packet_data = self.add_game_header(fragment, self.sequence);

            let packet_type = GamePacketType::select_by_size(packet_data.len(), &mut self.rng);
            let delay = self.compute_inter_packet_delay();

            morphed_packets.push(MorphedPacket {
                data: packet_data,
                delay,
                packet_type,
                sequence: self.sequence,
            });

            self.sequence = self.sequence.wrapping_add(1);
            self.fragment_state = frag_state;
        }

        tracing::debug!(
            output_packets = morphed_packets.len(),
            sequence_range_start = self.sequence.saturating_sub(morphed_packets.len() as u64),
            sequence_range_end = self.sequence,
            "Morphing complete"
        );

        morphed_packets
    }

    pub fn demorph(&mut self, packets: Vec<MorphedPacket>) -> Option<Vec<u8>> {
        if packets.is_empty() {
            return None;
        }

        let mut reassembled_fragments: Vec<Vec<u8>> = Vec::with_capacity(packets.len());
        let mut group_key: Option<u64> = None;

        for packet in packets {
            let (payload, seq) = match self.remove_game_header(&packet.data) {
                Some(result) => result,
                None => {
                    tracing::warn!(
                        sequence = packet.sequence,
                        "Failed to remove game header, skipping packet"
                    );
                    continue;
                }
            };

            if group_key.is_none() {
                group_key = Some(seq);
            }

            reassembled_fragments.push(payload);
        }

        if reassembled_fragments.is_empty() {
            return None;
        }

        let mut combined = Vec::new();
        for fragment in reassembled_fragments {
            combined.extend_from_slice(&fragment);
        }

        tracing::trace!(
            recovered_bytes = combined.len(),
            "Demorphing complete"
        );

        Some(combined)
    }

    // ─── Fragmentation ────────────────────────────────────────────────────

    fn fragment_data(&mut self, data: Vec<u8>) -> Vec<Vec<u8>> {
        let target_size = self.statistical_model.sample_packet_size(&mut self.rng);
        let payload_capacity = target_size.saturating_sub(HEADER_SIZE).max(MIN_PAYLOAD_SIZE);

        if data.len() <= payload_capacity {
            let padded = self.pad_to_target_size(data, target_size);
            return vec![padded];
        }

        let mut fragments = Vec::new();
        let mut remaining = data.as_slice();

        while !remaining.is_empty() {
            let current_target = self.statistical_model.sample_packet_size(&mut self.rng);
            let current_capacity = current_target.saturating_sub(HEADER_SIZE).max(MIN_PAYLOAD_SIZE);

            let chunk_size = remaining.len().min(current_capacity);
            let chunk = remaining[..chunk_size].to_vec();
            let padded = self.pad_to_target_size(chunk, current_target);
            fragments.push(padded);

            remaining = &remaining[chunk_size..];
        }

        tracing::trace!(
            total_fragments = fragments.len(),
            original_size = data.len(),
            avg_fragment_size = fragments.iter().map(|f| f.len()).sum::<usize>() / fragments.len().max(1),
            "Data fragmented"
        );

        fragments
    }

    // ─── Header Operations ────────────────────────────────────────────────

    fn add_game_header(&self, fragment: Vec<u8>, seq: u64) -> Vec<u8> {
        let total_size = HEADER_SIZE + fragment.len();
        let mut packet = vec![0u8; total_size];

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (d.as_millis() & 0xFFFFFFFF) as u32)
            .unwrap_or(0);

        let frag_info = self.fragment_state.encode();
        let original_len = fragment.len() as u32;
        let payload_len = fragment.len() as u32;

        packet[0] = MAGIC_BYTE;
        packet[1..4].copy_from_slice(&self.conversation_id.to_le_bytes());
        packet[4] = GamePacketType::select_by_size(fragment.len(), &mut {
            let mut rng = rand::thread_rng();
            rng
        })
        .to_byte();
        packet[5] = frag_info;
        packet[6..8].copy_from_slice(&32u16.to_le_bytes());
        packet[8..12].copy_from_slice(&now_ms.to_le_bytes());
        packet[12..16].copy_from_slice(&(seq as u32).to_le_bytes());
        packet[16..20].copy_from_slice(&original_len.to_le_bytes());
        packet[20..24].copy_from_slice(&payload_len.to_le_bytes());

        packet[HEADER_SIZE..].copy_from_slice(&fragment);

        packet
    }

    fn remove_game_header(&self, packet: &[u8]) -> Option<(Vec<u8>, u64)> {
        if packet.len() < HEADER_SIZE {
            tracing::warn!(
                packet_len = packet.len(),
                "Packet too short to contain game header"
            );
            return None;
        }

        if packet[0] != MAGIC_BYTE {
            tracing::warn!(
                magic_byte = packet[0],
                "Invalid magic byte in packet header"
            );
            return None;
        }

        let payload_len = u32::from_le_bytes([
            packet[20],
            packet[21],
            packet[22],
            packet[23],
        ]) as usize;

        let original_len = u32::from_le_bytes([
            packet[16],
            packet[17],
            packet[18],
            packet[19],
        ]) as usize;

        let seq = u32::from_le_bytes([
            packet[12],
            packet[13],
            packet[14],
            packet[15],
        ]) as u64;

        let expected_total = HEADER_SIZE + payload_len;
        if packet.len() < expected_total {
            tracing::warn!(
                expected = expected_total,
                actual = packet.len(),
                "Packet truncated"
            );
            return None;
        }

        let payload = packet[HEADER_SIZE..HEADER_SIZE + payload_len].to_vec();

        let frag_info = packet[5];
        let frag_state = FragmentState::decode(frag_info);

        if frag_state.total_fragments > 1 {
            let group_key = seq;
            let entry = self
                .reassembly_buffers
                .entry(group_key)
                .or_insert_with(|| ReassemblyBuffer::new(frag_state.total_fragments, original_len));

            entry.insert(frag_state.current_fragment, payload);

            if entry.is_complete() {
                let buffer = self.reassembly_buffers.remove(&group_key)?;
                return Some((buffer.reassemble(), seq));
            }
            return None;
        }

        Some((payload, seq))
    }

    // ─── Padding ──────────────────────────────────────────────────────────

    fn pad_to_target_size(&self, data: Vec<u8>, target_size: usize) -> Vec<u8> {
        let current_total = data.len();
        let payload_target = target_size.saturating_sub(HEADER_SIZE);

        if current_total >= payload_target {
            return data;
        }

        let padding_needed = payload_target - current_total;
        let mut result = data;
        result.reserve(padding_needed);

        let mut padding = vec![0u8; padding_needed];
        self.rng.fill(&mut padding[..]);

        let pad_marker_pos = padding_needed.saturating_sub(4);
        if padding_needed >= 4 {
            let pad_len = (padding_needed as u32).to_le_bytes();
            padding[pad_marker_pos..pad_marker_pos + 4].copy_from_slice(&pad_len);
        }

        result.extend_from_slice(&padding);

        tracing::trace!(
            original_size = current_total,
            padded_size = result.len(),
            padding_bytes = padding_needed,
            "Applied padding to match target size"
        );

        result
    }

    // ─── Oversized Splitting ──────────────────────────────────────────────

    fn split_oversized(&self, data: Vec<u8>, max_size: usize) -> Vec<Vec<u8>> {
        if data.len() <= max_size {
            return vec![data];
        }

        let chunk_size = max_size.saturating_sub(HEADER_SIZE).max(MIN_PAYLOAD_SIZE);
        let chunks: Vec<Vec<u8>> = data
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        tracing::trace!(
            original_size = data.len(),
            chunk_count = chunks.len(),
            chunk_size = chunk_size,
            "Split oversized data"
        );

        chunks
    }

    // ─── Small Packet Merging ─────────────────────────────────────────────

    fn merge_small_packets(&mut self) -> Vec<Vec<u8>> {
        if self.data_buffer.is_empty() {
            return Vec::new();
        }

        let target_size = self.statistical_model.sample_packet_size(&mut self.rng);
        let payload_capacity = target_size.saturating_sub(HEADER_SIZE).max(MIN_PAYLOAD_SIZE);

        if self.data_buffer.len() < payload_capacity / 2 {
            return Vec::new();
        }

        let mut merged = Vec::new();
        while self.data_buffer.len() >= payload_capacity {
            let chunk = self.data_buffer.drain(..payload_capacity).collect::<Vec<_>>();
            merged.push(chunk);
        }

        tracing::trace!(
            merged_packets = merged.len(),
            remaining_buffer = self.data_buffer.len(),
            "Merged small packets"
        );

        merged
    }

    // ─── Keepalive Generation ─────────────────────────────────────────────

    fn generate_keepalive_packets(&mut self) -> Vec<MorphedPacket> {
        let target_size = self.statistical_model.sample_packet_size(&mut self.rng);
        let payload_size = target_size.saturating_sub(HEADER_SIZE).max(MIN_PAYLOAD_SIZE);

        let mut keepalive_payload = vec![0u8; payload_size];
        self.rng.fill(&mut keepalive_payload[..]);

        let frag_state = FragmentState::new(0, 1, payload_size);
        self.fragment_state = frag_state;

        let packet_data = self.add_game_header(keepalive_payload, self.sequence);
        let delay = self.compute_inter_packet_delay();

        let packet = MorphedPacket {
            data: packet_data,
            delay,
            packet_type: GamePacketType::Keepalive,
            sequence: self.sequence,
        };

        self.sequence = self.sequence.wrapping_add(1);

        tracing::trace!(
            sequence = packet.sequence,
            size = packet.data.len(),
            "Generated keepalive packet"
        );

        vec![packet]
    }

    // ─── Delay Computation ────────────────────────────────────────────────

    fn compute_inter_packet_delay(&mut self) -> Duration {
        let base_iat = self.statistical_model.sample_inter_packet_time(&mut self.rng);

        let jitter = self.rng.gen_range(-3.0..=3.0);
        let jittered = (base_iat + jitter).max(1.0);

        let anti_periodicity = self.compute_anti_periodicity_offset();
        let adjusted = (jittered + anti_periodicity).max(1.0);

        Duration::from_millis(adjusted as u64)
    }

    fn compute_anti_periodicity_offset(&self) -> f64 {
        let std_iat = self.statistical_model.timing_dist.std_iat;
        let phase = (self.sequence as f64 * 0.137).sin();
        phase * std_iat * 0.15
    }

    // ─── Adversarial Integration ──────────────────────────────────────────

    pub fn apply_adversarial_decision(&mut self, data: &[u8]) -> MaskingDecision {
        self.adversarial_masker.decide(data)
    }

    pub fn get_feature_distance(&self) -> f64 {
        self.adversarial_masker.feature_distance()
    }

    // ─── Burst Engine Integration ─────────────────────────────────────────

    pub fn process_burst(&mut self, data: Vec<u8>) -> Vec<MorphedPacket> {
        let actions = self.burst_engine.process(data);
        let mut packets = Vec::new();

        for action in actions {
            match action {
                crate::masking::burst_engine::PacketAction::Send { data, delay } => {
                    let morphed = self.morph(data);
                    for mut p in morphed {
                        p.delay = delay.max(p.delay);
                        packets.push(p);
                    }
                }
                crate::masking::burst_engine::PacketAction::Pad { size, delay } => {
                    let mut padding = vec![0u8; size];
                    self.rng.fill(&mut padding[..]);
                    let morphed = self.morph(padding);
                    for mut p in morphed {
                        p.delay = delay.max(p.delay);
                        packets.push(p);
                    }
                }
                crate::masking::burst_engine::PacketAction::Split { data: chunks, delay } => {
                    for chunk in chunks {
                        let morphed = self.morph(chunk);
                        for mut p in morphed {
                            p.delay = delay.max(p.delay);
                            packets.push(p);
                        }
                    }
                }
                crate::masking::burst_engine::PacketAction::Keepalive { delay } => {
                    let morphed = self.generate_keepalive_packets();
                    for mut p in morphed {
                        p.delay = delay.max(p.delay);
                        packets.push(p);
                    }
                }
            }
        }

        packets
    }

    pub fn next_action_delay(&self) -> Duration {
        self.burst_engine.next_action_delay()
    }

    // ─── Accessors ────────────────────────────────────────────────────────

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn conversation_id(&self) -> u32 {
        self.conversation_id
    }

    pub fn buffered_data_len(&self) -> usize {
        self.data_buffer.len()
    }

    pub fn game_name(&self) -> &str {
        &self.statistical_model.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masking::profiles::{
        BurstPatternDef, EntropyProfileDef, GaussianComponentDef, GamingProfile, TimingBinDef,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_profile() -> GamingProfile {
        GamingProfile {
            name: "test_game".to_string(),
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
                    mean: 350.0,
                    std_dev: 40.0,
                    weight: 0.20,
                },
            ],
            timing_distribution: vec![
                TimingBinDef {
                    time_ms: 16.67,
                    probability: 0.65,
                },
                TimingBinDef {
                    time_ms: 33.33,
                    probability: 0.20,
                },
                TimingBinDef {
                    time_ms: 50.0,
                    probability: 0.15,
                },
            ],
            burst_patterns: vec![
                BurstPatternDef {
                    name: "movement".to_string(),
                    packet_count_range: (2, 5),
                    inter_packet_time_range: (10.0, 25.0),
                    size_range: (60, 120),
                    probability: 0.6,
                },
                BurstPatternDef {
                    name: "shooting".to_string(),
                    packet_count_range: (3, 8),
                    inter_packet_time_range: (5.0, 15.0),
                    size_range: (80, 180),
                    probability: 0.4,
                },
            ],
            client_server_size_ratio: (0.5, 1.0),
            entropy_profile: Some(EntropyProfileDef {
                mean_entropy: 7.2,
                entropy_std: 0.3,
            }),
        }
    }

    fn make_morpher() -> TrafficMorpher {
        let profile = test_profile();
        let model = StatisticalGameTrafficModel::from_profile(&profile).unwrap();
        TrafficMorpher::new(model).unwrap()
    }

    #[test]
    fn test_morpher_creation() {
        let morpher = make_morpher();
        assert_eq!(morpher.sequence(), 0);
        assert_eq!(morpher.buffered_data_len(), 0);
        assert_eq!(morpher.game_name(), "test_game");
    }

    #[test]
    fn test_morph_small_data_produces_packets() {
        let mut morpher = make_morpher();
        let data = vec![0xDEu8; 100];
        let packets = morpher.morph(data);

        assert!(!packets.is_empty());
        for packet in &packets {
            assert!(packet.data.len() >= HEADER_SIZE);
            assert!(packet.data[0] == MAGIC_BYTE);
        }
    }

    #[test]
    fn test_morph_large_data_produces_fragments() {
        let mut morpher = make_morpher();
        let data = vec![0xABu8; 5000];
        let packets = morpher.morph(data);

        assert!(packets.len() > 1);
        for packet in &packets {
            assert!(packet.data.len() >= HEADER_SIZE);
            assert!(packet.data[0] == MAGIC_BYTE);
        }
    }

    #[test]
    fn test_morph_empty_data_produces_keepalive() {
        let mut morpher = make_morpher();
        let packets = morpher.morph(Vec::new());

        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].packet_type, GamePacketType::Keepalive);
    }

    #[test]
    fn test_demorph_roundtrip_small() {
        let mut morpher = make_morpher();
        let original = vec![0x42u8; 128];
        let packets = morpher.morph(original.clone());

        let recovered = morpher.demorph(packets);
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap(), original);
    }

    #[test]
    fn test_demorph_roundtrip_large() {
        let mut morpher = make_morpher();
        let mut original = vec![0u8; 10000];
        let mut rng = StdRng::seed_from_u64(42);
        rng.fill(&mut original[..]);

        let packets = morpher.morph(original.clone());
        let recovered = morpher.demorph(packets);

        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap(), original);
    }

    #[test]
    fn test_demorph_roundtrip_binary_data() {
        let mut morpher = make_morpher();
        let original: Vec<u8> = (0..255).cycle().take(2048).collect();
        let packets = morpher.morph(original.clone());
        let recovered = morpher.demorph(packets);

        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap(), original);
    }

    #[test]
    fn test_no_data_loss_across_multiple_morph_calls() {
        let mut morpher = make_morpher();
        let datasets: Vec<Vec<u8>> = (0..10)
            .map(|i| vec![i as u8; 500 + i * 100])
            .collect();

        for original in datasets {
            let packets = morpher.morph(original.clone());
            let recovered = morpher.demorph(packets);
            assert!(recovered.is_some(), "Data loss detected");
            assert_eq!(recovered.unwrap(), original);
        }
    }

    #[test]
    fn test_packet_types_are_varied() {
        let mut morpher = make_morpher();
        let data = vec![0u8; 3000];
        let packets = morpher.morph(data);

        let types: Vec<_> = packets.iter().map(|p| p.packet_type).collect();
        assert!(!types.is_empty());
    }

    #[test]
    fn test_sequence_numbers_increment() {
        let mut morpher = make_morpher();
        let data = vec![0u8; 2000];
        let packets = morpher.morph(data);

        for i in 1..packets.len() {
            assert!(packets[i].sequence > packets[i - 1].sequence);
        }
    }

    #[test]
    fn test_delays_are_nonzero() {
        let mut morpher = make_morpher();
        let data = vec![0u8; 500];
        let packets = morpher.morph(data);

        for packet in &packets {
            assert!(packet.delay > Duration::from_secs(0));
        }
    }

    #[test]
    fn test_delays_are_reasonable() {
        let mut morpher = make_morpher();
        let data = vec![0u8; 500];
        let packets = morpher.morph(data);

        for packet in &packets {
            assert!(packet.delay < Duration::from_secs(5));
        }
    }

    #[test]
    fn test_fragment_state_encoding_decoding() {
        let state = FragmentState::new(2, 5, 128);
        let encoded = state.encode();
        let decoded = FragmentState::decode(encoded);

        assert_eq!(decoded.current_fragment, 2);
        assert_eq!(decoded.total_fragments, 5);
    }

    #[test]
    fn test_fragment_state_single_fragment() {
        let state = FragmentState::new(0, 1, 256);
        let encoded = state.encode();
        let decoded = FragmentState::decode(encoded);

        assert_eq!(decoded.current_fragment, 0);
        assert_eq!(decoded.total_fragments, 1);
    }

    #[test]
    fn test_remove_header_invalid_magic() {
        let morpher = make_morpher();
        let bad_packet = vec![0x00u8; 32];
        let result = morpher.remove_game_header(&bad_packet);
        assert!(result.is_none());
    }

    #[test]
    fn test_remove_header_too_short() {
        let morpher = make_morpher();
        let short_packet = vec![MAGIC_BYTE, 0, 0, 0];
        let result = morpher.remove_game_header(&short_packet);
        assert!(result.is_none());
    }

    #[test]
    fn test_pad_to_target_size() {
        let morpher = make_morpher();
        let data = vec![0xFFu8; 50];
        let padded = morpher.pad_to_target_size(data.clone(), 200);

        assert!(padded.len() >= 50);
        assert_eq!(&padded[..50], &data[..]);
    }

    #[test]
    fn test_pad_no_op_when_already_large() {
        let morpher = make_morpher();
        let data = vec![0xFFu8; 500];
        let padded = morpher.pad_to_target_size(data.clone(), 200);

        assert_eq!(padded, data);
    }

    #[test]
    fn test_split_oversized() {
        let morpher = make_morpher();
        let data = vec![0xCCu8; 3000];
        let chunks = morpher.split_oversized(data.clone(), 500);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 500);
        }

        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn test_split_no_op_when_under_size() {
        let morpher = make_morpher();
        let data = vec![0xCCu8; 100];
        let chunks = morpher.split_oversized(data.clone(), 500);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], data);
    }

    #[test]
    fn test_feature_distance_is_nonnegative() {
        let morpher = make_morpher();
        let distance = morpher.get_feature_distance();
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_process_burst_with_data() {
        let mut morpher = make_morpher();
        let packets = morpher.process_burst(vec![0x11u8; 200]);

        assert!(!packets.is_empty());
        for packet in &packets {
            assert!(packet.data[0] == MAGIC_BYTE);
        }
    }

    #[test]
    fn test_next_action_delay_returns_valid() {
        let morpher = make_morpher();
        let delay = morpher.next_action_delay();
        assert!(delay > Duration::from_secs(0));
        assert!(delay < Duration::from_secs(30));
    }

    #[test]
    fn test_conversation_id_is_stable() {
        let morpher = make_morpher();
        let cid1 = morpher.conversation_id();
        let cid2 = morpher.conversation_id();
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn test_morph_demorph_exact_byte_preservation() {
        let mut morpher = make_morpher();
        let original: Vec<u8> = (0..256u8)
            .flat_map(|b| std::iter::repeat(b).take(17))
            .collect();

        let packets = morpher.morph(original.clone());
        let recovered = morpher.demorph(packets);

        assert!(recovered.is_some());
        let recovered = recovered.unwrap();
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered, original);
    }
}
