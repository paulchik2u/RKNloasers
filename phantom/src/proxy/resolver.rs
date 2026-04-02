use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, trace};

use crate::masking::kcp_wrapper::KcpWrapper;
use crate::transport::QuicClient;
use crate::utils::error::{PhantomError, PhantomResult};

use super::socks5::{Socks5Address, Socks5Request};

const DNS_RESPONSE_TIMEOUT_MS: u64 = 10000;

pub struct TunnelResolver {
    quic_client: Arc<QuicClient>,
    kcp_wrapper: Arc<Mutex<KcpWrapper>>,
}

impl TunnelResolver {
    pub fn new(quic_client: Arc<QuicClient>, kcp_wrapper: Arc<Mutex<KcpWrapper>>) -> Self {
        info!("TunnelResolver initialized");
        Self {
            quic_client,
            kcp_wrapper,
        }
    }

    pub async fn resolve(&self, request: Socks5Request) -> PhantomResult<Vec<u8>> {
        debug!(
            "Resolving target through tunnel: {}:{}, cmd={}",
            request.target.host_str(),
            request.target.port(),
            request.command
        );

        let target_bytes = encode_target_address(&request.target);

        let mut kcp = self.kcp_wrapper.lock().await;
        kcp.send(&target_bytes)
            .await
            .map_err(|e| PhantomError::KcpError(format!("Failed to send through KCP tunnel: {}", e)))?;

        trace!("Sent target address through KCP tunnel");

        let response = kcp
            .recv()
            .await
            .map_err(|e| PhantomError::KcpError(format!("Failed to receive from KCP tunnel: {}", e)))?;

        trace!("Received {} bytes from tunnel", response.len());

        Ok(response)
    }

    pub async fn resolve_dns(&self, domain: &str) -> PhantomResult<Vec<u8>> {
        debug!("Resolving DNS through tunnel: {}", domain);

        let dns_query = build_dns_query(domain)?;

        let mut tunnel_payload = Vec::with_capacity(3 + dns_query.len());
        tunnel_payload.push(0x00);
        tunnel_payload.push(0x00);
        tunnel_payload.push(0x01);
        tunnel_payload.extend_from_slice(&dns_query);

        let mut kcp = self.kcp_wrapper.lock().await;
        kcp.send(&tunnel_payload)
            .await
            .map_err(|e| PhantomError::KcpError(format!("DNS query send failed: {}", e)))?;

        let response = tokio::time::timeout(
            std::time::Duration::from_millis(DNS_RESPONSE_TIMEOUT_MS),
            kcp.recv(),
        )
        .await
        .map_err(|_| PhantomError::ConnectionFailed("DNS resolution timed out through tunnel".to_string()))?
        .map_err(|e| PhantomError::KcpError(format!("DNS recv failed: {}", e)))?;

        debug!("DNS response received: {} bytes", response.len());
        Ok(response)
    }

    pub async fn forward_stream(&self, data: &[u8]) -> PhantomResult<Vec<u8>> {
        trace!("Forwarding {} bytes through tunnel", data.len());

        let mut kcp = self.kcp_wrapper.lock().await;
        kcp.send(data)
            .await
            .map_err(|e| PhantomError::KcpError(format!("Forward send failed: {}", e)))?;

        let response = kcp
            .recv()
            .await
            .map_err(|e| PhantomError::KcpError(format!("Forward recv failed: {}", e)))?;

        trace!("Received {} bytes from tunnel", response.len());
        Ok(response)
    }
}

fn encode_target_address(target: &Socks5Address) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    match target {
        Socks5Address::Ipv4(ip, port) => {
            buf.push(0x01);
            buf.extend_from_slice(&ip.octets());
            buf.extend_from_slice(&port.to_be_bytes());
        }
        Socks5Address::Ipv6(ip, port) => {
            buf.push(0x04);
            buf.extend_from_slice(&ip.octets());
            buf.extend_from_slice(&port.to_be_bytes());
        }
        Socks5Address::Domain(domain, port) => {
            buf.push(0x03);
            buf.push(domain.len() as u8);
            buf.extend_from_slice(domain.as_bytes());
            buf.extend_from_slice(&port.to_be_bytes());
        }
    }
    buf
}

fn build_dns_query(domain: &str) -> PhantomResult<Vec<u8>> {
    let mut query = Vec::with_capacity(domain.len() + 16);

    let txid: u16 = rand::random();
    query.extend_from_slice(&txid.to_be_bytes());

    query.extend_from_slice(&[
        0x01, 0x00,
        0x00, 0x01,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
    ]);

    for label in domain.split('.') {
        if label.is_empty() {
            continue;
        }
        let bytes = label.as_bytes();
        query.push(bytes.len() as u8);
        query.extend_from_slice(bytes);
    }

    query.push(0x00);

    query.extend_from_slice(&[
        0x00, 0x01,
        0x00, 0x01,
    ]);

    trace!("Built DNS query for {}: {} bytes", domain, query.len());
    Ok(query)
}
