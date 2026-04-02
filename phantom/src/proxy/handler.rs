use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};

use crate::utils::error::{PhantomError, PhantomResult};

use super::resolver::TunnelResolver;
use super::socks5::{
    read_address, send_reply, ConnectionState, read_exact_with_timeout,
    Socks5Address, Socks5Request,
};

const SOCKS5_VERSION: u8 = 0x05;
const NO_AUTH: u8 = 0x00;
const CMD_CONNECT: u8 = 0x01;
const CMD_BIND: u8 = 0x02;
const CMD_UDP_ASSOCIATE: u8 = 0x03;
const REPLY_SUCCEEDED: u8 = 0x00;
const REPLY_COMMAND_UNSUPPORTED: u8 = 0x07;
const REPLY_GENERAL_FAILURE: u8 = 0x01;

pub struct ConnectionHandler {
    resolver: Arc<TunnelResolver>,
    state: Arc<RwLock<ConnectionState>>,
}

impl ConnectionHandler {
    pub fn new(resolver: Arc<TunnelResolver>, state: Arc<RwLock<ConnectionState>>) -> Self {
        Self { resolver, state }
    }

    pub async fn handle(&self, mut stream: TcpStream) -> PhantomResult<()> {
        let peer = stream.peer_addr().ok().map(|a| a.to_string()).unwrap_or_else(|| "unknown".to_string());
        debug!("Handling new SOCKS5 connection from {}", peer);

        match self.negotiate_auth(&mut stream).await {
            Ok(()) => trace!("Auth negotiation successful for {}", peer),
            Err(e) => {
                warn!("Auth negotiation failed for {}: {}", peer, e);
                return Err(e);
            }
        }

        let request = match self.read_request(&mut stream).await {
            Ok(req) => {
                debug!("Received SOCKS5 request from {}: cmd={}, target={}:{}", 
                    peer, req.command, req.target.host_str(), req.target.port());
                req
            }
            Err(e) => {
                warn!("Failed to read SOCKS5 request from {}: {}", peer, e);
                return Err(e);
            }
        };

        match request.command {
            CMD_CONNECT => self.handle_connect(&mut stream, &request).await,
            CMD_BIND => self.handle_bind(&mut stream).await,
            CMD_UDP_ASSOCIATE => self.handle_udp_associate(&mut stream).await,
            _ => {
                warn!("Unknown SOCKS5 command: {} from {}", request.command, peer);
                send_reply(&mut stream, REPLY_COMMAND_UNSUPPORTED, &Socks5Address::Ipv4([0,0,0,0].into(), 0)).await?;
                Err(PhantomError::ConnectionLost(format!("Unsupported command: {}", request.command)))
            }
        }
    }

    async fn negotiate_auth(&self, stream: &mut TcpStream) -> PhantomResult<()> {
        let mut buf = [0u8; 2];
        read_exact_with_timeout(stream, &mut buf).await?;

        let version = buf[0];
        let nmethods = buf[1] as usize;

        if version != SOCKS5_VERSION {
            return Err(PhantomError::ConnectionLost(format!("Unsupported SOCKS version: {}", version)));
        }

        trace!("Auth negotiation: version={}, nmethods={}", version, nmethods);

        let mut methods = vec![0u8; nmethods];
        read_exact_with_timeout(stream, &mut methods).await?;

        let supports_no_auth = methods.iter().any(|&m| m == NO_AUTH);

        if supports_no_auth {
            stream.write_all(&[SOCKS5_VERSION, NO_AUTH]).await
                .map_err(PhantomError::IoError)?;
            trace!("Selected NO AUTH method");
            Ok(())
        } else {
            stream.write_all(&[SOCKS5_VERSION, 0xFF]).await
                .map_err(PhantomError::IoError)?;
            Err(PhantomError::ConnectionLost("No acceptable auth methods".to_string()))
        }
    }

    async fn read_request(&self, stream: &mut TcpStream) -> PhantomResult<Socks5Request> {
        let mut header = [0u8; 3];
        read_exact_with_timeout(stream, &mut header).await?;

        let version = header[0];
        let command = header[1];
        let _reserved = header[2];

        if version != SOCKS5_VERSION {
            return Err(PhantomError::ConnectionLost(format!("Invalid request version: {}", version)));
        }

        let target = read_address(stream).await?;

        Ok(Socks5Request { command, target })
    }

    async fn handle_connect(&self, stream: &mut TcpStream, request: &Socks5Request) -> PhantomResult<()> {
        info!(
            "CONNECT request: {}:{} ",
            request.target.host_str(),
            request.target.port()
        );

        let tunnel_response = self.resolver.resolve(request.clone()).await;

        match tunnel_response {
            Ok(response) => {
                if response.is_empty() || response[0] != 0x00 {
                    warn!("Tunnel resolution failed for {}:{} ",
                        request.target.host_str(), request.target.port());
                    send_reply(stream, REPLY_GENERAL_FAILURE, &request.target).await?;
                    return Err(PhantomError::ConnectionLost("Tunnel resolution failed".to_string()));
                }

                send_reply(stream, REPLY_SUCCEEDED, &request.target).await?;

                self.relay_traffic(stream, &response[1..]).await
            }
            Err(e) => {
                error!("Tunnel resolution error: {}", e);
                send_reply(stream, REPLY_GENERAL_FAILURE, &request.target).await?;
                Err(e)
            }
        }
    }

    async fn handle_bind(&self, stream: &mut TcpStream) -> PhantomResult<()> {
        warn!("BIND command not supported");
        send_reply(stream, REPLY_COMMAND_UNSUPPORTED, &Socks5Address::Ipv4([0,0,0,0].into(), 0)).await?;
        Err(PhantomError::ConnectionLost("BIND command not supported".to_string()))
    }

    async fn handle_udp_associate(&self, stream: &mut TcpStream) -> PhantomResult<()> {
        warn!("UDP ASSOCIATE command not supported");
        send_reply(stream, REPLY_COMMAND_UNSUPPORTED, &Socks5Address::Ipv4([0,0,0,0].into(), 0)).await?;
        Err(PhantomError::ConnectionLost("UDP ASSOCIATE not supported".to_string()))
    }

    async fn relay_traffic(&self, client_stream: &mut TcpStream, initial_data: &[u8]) -> PhantomResult<()> {
        trace!("Starting traffic relay, initial data: {} bytes", initial_data.len());

        let mut buf = [0u8; 8192];
        let mut total_in: u64 = 0;
        let mut total_out: u64 = 0;

        if !initial_data.is_empty() {
            let response = self.resolver.forward_stream(initial_data).await?;
            client_stream.write_all(&response).await.map_err(PhantomError::IoError)?;
            total_in += initial_data.len() as u64;
            total_out += response.len() as u64;
        }

        loop {
            let n = client_stream.read(&mut buf).await.map_err(PhantomError::IoError)?;
            if n == 0 {
                trace!("Client stream closed, ending relay");
                break;
            }

            let data = &buf[..n];
            total_in += n as u64;

            let response = self.resolver.forward_stream(data).await?;
            total_out += response.len() as u64;

            if !response.is_empty() {
                client_stream.write_all(&response).await.map_err(PhantomError::IoError)?;
            }
        }

        {
            let mut state = self.state.write().await;
            state.total_bytes_in += total_in;
            state.total_bytes_out += total_out;
        }

        trace!("Traffic relay complete: in={} bytes, out={} bytes", total_in, total_out);
        Ok(())
    }
}
