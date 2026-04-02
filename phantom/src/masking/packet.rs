use rand::Rng;

use crate::masking::profiles::GamingProfile;

const KCP_HEADER_SIZE: usize = 24;
const KCP_HEADER_LEN_OFFSET: usize = 20;

pub struct PacketBuilder {
    profile: GamingProfile,
    sequence: u32,
}

impl PacketBuilder {
    pub fn new(profile: GamingProfile) -> Self {
        tracing::info!(
            profile = %profile.name,
            size_range = %format!("{}-{}", profile.packet_size_min, profile.packet_size_max),
            "Initialized PacketBuilder"
        );
        PacketBuilder {
            profile,
            sequence: 0,
        }
    }

    pub fn build(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        if data.is_empty() {
            return Vec::new();
        }

        let max_payload = self.max_payload_size();
        let chunk_count = (data.len() + max_payload - 1) / max_payload;
        let mut packets = Vec::with_capacity(chunk_count);

        for (i, chunk) in data.chunks(max_payload).enumerate() {
            let is_last = i == chunk_count - 1;
            let packet = self.build_kcp_packet(chunk, i, is_last);
            packets.push(packet);
        }

        self.sequence += chunk_count as u32;
        tracing::debug!(
            input_bytes = data.len(),
            packet_count = packets.len(),
            "Built game-sized packets"
        );
        packets
    }

    pub fn pad_packet(&self, packet: &mut Vec<u8>) {
        let target_size = self.random_packet_size();
        let current_len = packet.len();

        if current_len >= target_size {
            return;
        }

        let padding_needed = target_size - current_len;
        let mut rng = rand::thread_rng();
        let mut padding: Vec<u8> = vec![0u8; padding_needed];
        rng.fill(&mut padding[..]);

        packet.extend_from_slice(&padding);

        let total_len = packet.len() as u32;
        packet[KCP_HEADER_LEN_OFFSET..KCP_HEADER_LEN_OFFSET + 4]
            .copy_from_slice(&total_len.to_le_bytes());

        tracing::trace!(
            original_size = current_len,
            padded_size = packet.len(),
            "Padded packet to match game traffic profile"
        );
    }

    pub fn profile(&self) -> &GamingProfile {
        &self.profile
    }

    fn max_payload_size(&self) -> usize {
        let max_packet = self.profile.packet_size_max as usize;
        if max_packet > KCP_HEADER_SIZE {
            max_packet - KCP_HEADER_SIZE
        } else {
            64
        }
    }

    fn build_kcp_packet(&self, payload: &[u8], fragment_index: usize, is_last: bool) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let conv: u32 = rng.gen_range(1..=u32::MAX);
        let cmd: u8 = 0x81;
        let total_fragments = ((payload.len() + self.max_payload_size() - 1) / self.max_payload_size()).max(1) as u8;
        let frg: u8 = if is_last { 0 } else { (total_fragments - 1 - fragment_index as u8) & 0xFF };
        let wnd: u16 = 32;
        let ts: u32 = rng.gen();
        let sn: u32 = self.sequence + fragment_index as u32;
        let una: u32 = self.sequence;
        let len: u32 = (KCP_HEADER_SIZE + payload.len()) as u32;

        let mut packet = Vec::with_capacity(len as usize);
        packet.extend_from_slice(&conv.to_le_bytes());
        packet.push(cmd);
        packet.push(frg);
        packet.extend_from_slice(&wnd.to_le_bytes());
        packet.extend_from_slice(&ts.to_le_bytes());
        packet.extend_from_slice(&sn.to_le_bytes());
        packet.extend_from_slice(&una.to_le_bytes());
        packet.extend_from_slice(&len.to_le_bytes());
        packet.extend_from_slice(payload);

        packet
    }

    fn random_packet_size(&self) -> usize {
        let mut rng = rand::thread_rng();
        rng.gen_range(self.profile.packet_size_min..=self.profile.packet_size_max) as usize
    }

    pub fn parse_kcp_packets(data: &[u8]) -> Vec<Vec<u8>> {
        let mut packets = Vec::new();
        let mut offset = 0;

        while offset + KCP_HEADER_SIZE <= data.len() {
            if offset + KCP_HEADER_LEN_OFFSET + 4 > data.len() {
                break;
            }

            let len_bytes: [u8; 4] = data[offset + KCP_HEADER_LEN_OFFSET..offset + KCP_HEADER_LEN_OFFSET + 4]
                .try_into()
                .unwrap_or([0u8; 4]);
            let packet_len = u32::from_le_bytes(len_bytes) as usize;

            if packet_len < KCP_HEADER_SIZE || offset + packet_len > data.len() {
                break;
            }

            packets.push(data[offset..offset + packet_len].to_vec());
            offset += packet_len;
        }

        packets
    }
}
