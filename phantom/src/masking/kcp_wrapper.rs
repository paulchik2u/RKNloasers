use kcp::Kcp;
use rand::Rng;
use std::io::Write;
use std::sync::{Arc, RwLock};

use crate::core::state::ConnectionState;
use crate::masking::packet::PacketBuilder;
use crate::masking::profiles::GamingProfile;
use crate::masking::timing::TrafficShaper;
use crate::utils::{PhantomError, PhantomResult};

#[derive(Debug, Default)]
pub struct KcpOutput {
    buffer: Vec<u8>,
}

impl KcpOutput {
    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buffer)
    }
}

impl Write for KcpOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct KcpWrapper {
    kcp: Kcp<KcpOutput>,
    packet_builder: PacketBuilder,
    traffic_shaper: TrafficShaper,
    state: Arc<RwLock<ConnectionState>>,
    conv: u32,
}

impl KcpWrapper {
    pub fn new(profile: GamingProfile, state: Arc<RwLock<ConnectionState>>) -> PhantomResult<Self> {
        let mut rng = rand::thread_rng();
        let conv: u32 = rng.gen_range(1..=u32::MAX);

        let output = KcpOutput::default();
        let mut kcp = Kcp::new(conv, output);
        kcp.set_wndsize(32, 32);
        kcp.set_nodelay(true, 20, 2, true);
        kcp.set_mtu(1400).map_err(|e| {
            tracing::error!(error = %e, "Failed to set KCP MTU");
            PhantomError::KcpError(format!("Failed to set MTU: {}", e))
        })?;

        let packet_builder = PacketBuilder::new(profile.clone());
        let traffic_shaper = TrafficShaper::new(profile);

        tracing::info!(
            conv = conv,
            snd_wnd = 32,
            rcv_wnd = 32,
            "Initialized KcpWrapper"
        );

        Ok(KcpWrapper {
            kcp,
            packet_builder,
            traffic_shaper,
            state,
            conv,
        })
    }

    pub async fn send(&mut self, data: &[u8]) -> PhantomResult<Vec<Vec<u8>>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let bytes_written = self.kcp.send(data).map_err(|e| {
            tracing::error!(error = %e, "KCP send failed");
            PhantomError::KcpError(format!("KCP send error: {}", e))
        })?;

        tracing::debug!(bytes_written = bytes_written, "Data queued in KCP send buffer");

        self.kcp.flush().map_err(|e| {
            tracing::error!(error = %e, "KCP flush failed");
            PhantomError::KcpError(format!("KCP flush error: {}", e))
        })?;

        let raw_output = self.kcp.get_output_mut().take();
        let packets = PacketBuilder::parse_kcp_packets(&raw_output);

        let mut final_packets = Vec::with_capacity(packets.len());
        for mut packet in packets {
            self.packet_builder.pad_packet(&mut packet);
            final_packets.push(packet);
        }

        if self.traffic_shaper.should_send_heartbeat() {
            let heartbeat = self.traffic_shaper.heartbeat_packet();
            final_packets.push(heartbeat);
            self.traffic_shaper.mark_heartbeat_sent();
            tracing::trace!("Appended heartbeat packet to outgoing batch");
        }

        tracing::debug!(
            packet_count = final_packets.len(),
            "Prepared packets for transmission"
        );

        Ok(final_packets)
    }

    pub async fn recv(&mut self, packet: &[u8]) -> PhantomResult<Vec<u8>> {
        if packet.len() < 24 {
            return Err(PhantomError::KcpError(
                "Received packet too small to be a valid KCP packet".to_string(),
            ));
        }

        self.kcp.input(packet).map_err(|e| {
            tracing::error!(error = %e, "KCP input failed");
            PhantomError::KcpError(format!("KCP input error: {}", e))
        })?;

        match self.kcp.peeksize() {
            Ok(size) if size > 0 => {
                let mut buf = vec![0u8; size];
                let bytes_read = self.kcp.recv(&mut buf).map_err(|e| {
                    tracing::error!(error = %e, "KCP recv failed");
                    PhantomError::KcpError(format!("KCP recv error: {}", e))
                })?;
                buf.truncate(bytes_read);
                tracing::debug!(bytes_read = bytes_read, "Decapsulated KCP data");
                Ok(buf)
            }
            Ok(_) => {
                Ok(Vec::new())
            }
            Err(e) => {
                tracing::warn!(error = %e, "KCP peeksize failed");
                Err(PhantomError::KcpError(format!("KCP peeksize error: {}", e)))
            }
        }
    }

    pub fn update(&mut self, current_ms: u32) {
        if let Err(e) = self.kcp.update(current_ms) {
            tracing::warn!(error = %e, current_ms = current_ms, "KCP update failed");
        }
    }

    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        if let Err(e) = self.kcp.flush() {
            tracing::warn!(error = %e, "KCP flush failed during manual flush");
            return Vec::new();
        }

        let raw_output = self.kcp.get_output_mut().take();
        let packets = PacketBuilder::parse_kcp_packets(&raw_output);

        let mut final_packets = Vec::with_capacity(packets.len());
        for mut packet in packets {
            self.packet_builder.pad_packet(&mut packet);
            final_packets.push(packet);
        }

        final_packets
    }

    pub fn conv(&self) -> u32 {
        self.conv
    }

    pub fn waiting_conv(&self) -> bool {
        self.kcp.waiting_conv()
    }

    pub fn is_dead_link(&self) -> bool {
        self.kcp.is_dead_link()
    }

    pub fn wait_snd(&self) -> usize {
        self.kcp.wait_snd()
    }
}
