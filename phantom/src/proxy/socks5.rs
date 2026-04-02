use crate::utils::error::{PhantomError, PhantomResult};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};

use super::handler::ConnectionHandler;
use super::resolver::TunnelResolver;

const SOCKS5_VERSION: u8 = 0x05;
const NO_AUTH: u8 = 0x00;
const CMD_CONNECT: u8 = 0x01;
const CMD_BIND: u8 = 0x02;
const CMD_UDP_ASSOCIATE: u8 = 0x03;
const ADDR_TYPE_IPV4: u8 = 0x01;
const ADDR_TYPE_DOMAIN: u8 = 0x03;
const ADDR_TYPE_IPV6: u8 = 0x04;
const REPLY_SUCCEEDED: u8 = 0x00;
const REPLY_COMMAND_UNSUPPORTED: u8 = 0x07;
const REPLY_GENERAL_FAILURE: u8 = 0x01;

#[derive(Debug, Clone)]
pub enum Socks5Address {
    Ipv4(Ipv4Addr, u16),
    Ipv6(Ipv6Addr, u16),
    Domain(String, u16),
}

impl Socks5Address {
    pub fn host_str(&self) -> String {
        match self {
            Socks5Address::Ipv4(ip, _) => ip.to_string(),
            Socks5Address::Ipv6(ip, _) => ip.to_string(),
            Socks5Address::Domain(domain, _) => domain.clone(),
        }
    }

    pub fn port(&self) -> u16 {
        match self {
            Socks5Address::Ipv4(_, port) => *port,
            Socks5Address::Ipv6(_, port) => *port,
            Socks5Address::Domain(_, port) => *port,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Socks5Request {
    pub command: u8,
    pub target: Socks5Address,
}

impl Socks5Request {
    pub fn is_connect(&self) -> bool {
        self.command == CMD_CONNECT
    }

    pub fn is_bind(&self) -> bool {
        self.command == CMD_BIND
    }

    pub fn is_udp_associate(&self) -> bool {
        self.command == CMD_UDP_ASSOCIATE
    }
}

pub struct ConnectionState {
    pub active_connections: u64,
    pub total_bytes_in: u64,
    pub total_bytes_out: u64,
    pub is_shutting_down: bool,
}

impl ConnectionState {
    pub fn new() -> Self {
        Self {
            active_connections: 0,
            total_bytes_in: 0,
            total_bytes_out: 0,
            is_shutting_down: false,
        }
    }
}

pub struct Socks5Proxy {
    listener: Arc<RwLock<Option<tokio::net::TcpListener>>>,
    handler: Arc<ConnectionHandler>,
    port: u16,
    state: Arc<RwLock<ConnectionState>>,
}

impl Socks5Proxy {
    pub fn new(port: u16, resolver: Arc<TunnelResolver>, state: Arc<RwLock<ConnectionState>>) -> Self {
        let handler = Arc::new(ConnectionHandler::new(resolver, state.clone()));
        Self {
            listener: Arc::new(RwLock::new(None)),
            handler,
            port,
            state,
        }
    }

    pub async fn start(&self) -> PhantomResult<()> {
        let bind_addr = format!("127.0.0.1:{}", self.port);
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| PhantomError::IoError(std::io::Error::new(e.kind(), format!("Failed to bind SOCKS5 proxy to {}: {}", bind_addr, e))))?;

        info!("SOCKS5 proxy listening on {}", bind_addr);

        {
            let mut guard = self.listener.write().await;
            *guard = Some(listener);
        }

        let listener = {
            let guard = self.listener.read().await;
            guard.clone()
        };

        if let Some(listener) = listener {
            loop {
                {
                    let state = self.state.read().await;
                    if state.is_shutting_down {
                        info!("SOCKS5 proxy shutdown detected, stopping accept loop");
                        break;
                    }
                }

                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        debug!("Accepted SOCKS5 connection from {}", peer_addr);
                        let handler = self.handler.clone();
                        let state = self.state.clone();

                        tokio::spawn(async move {
                            {
                                let mut state_guard = state.write().await;
                                state_guard.active_connections += 1;
                            }

                            match handler.handle(stream).await {
                                Ok(()) => debug!("Connection from {} handled successfully", peer_addr),
                                Err(e) => warn!("Connection from {} failed: {}", peer_addr, e),
                            }

                            {
                                let mut state_guard = state.write().await;
                                state_guard.active_connections -= 1;
                            }
                        });
                    }
                    Err(e) => {
                        let should_break = {
                            let state = self.state.read().await;
                            state.is_shutting_down
                        };
                        if should_break {
                            info!("SOCKS5 proxy shutting down, exiting accept loop");
                            break;
                        }
                        error!("Failed to accept SOCKS5 connection: {}", e);
                    }
                }
            }
        }

        info!("SOCKS5 proxy accept loop exited");
        Ok(())
    }

    pub async fn stop(&self) -> PhantomResult<()> {
        info!("Stopping SOCKS5 proxy");

        {
            let mut state = self.state.write().await;
            state.is_shutting_down = true;
        }

        {
            let mut guard = self.listener.write().await;
            if let Some(listener) = guard.take() {
                drop(listener);
                debug!("SOCKS5 listener dropped");
            }
        }

        info!("SOCKS5 proxy stopped");
        Ok(())
    }

    pub async fn get_active_connections(&self) -> u64 {
        let state = self.state.read().await;
        state.active_connections
    }
}

pub async fn read_exact_with_timeout(stream: &mut TcpStream, buf: &mut [u8]) -> PhantomResult<()> {
    let mut total_read = 0;
    while total_read < buf.len() {
        let n = stream
            .read(&mut buf[total_read..])
            .await
            .map_err(PhantomError::IoError)?;
        if n == 0 {
            return Err(PhantomError::ConnectionLost("Connection closed during read".to_string()));
        }
        total_read += n;
    }
    Ok(())
}

pub async fn read_address(stream: &mut TcpStream) -> PhantomResult<Socks5Address> {
    let mut addr_type_buf = [0u8; 1];
    read_exact_with_timeout(stream, &mut addr_type_buf).await?;
    let addr_type = addr_type_buf[0];

    trace!("Reading SOCKS5 address type: {}", addr_type);

    match addr_type {
        ADDR_TYPE_IPV4 => {
            let mut buf = [0u8; 6];
            read_exact_with_timeout(stream, &mut buf).await?;
            let ip = Ipv4Addr::new(buf[0], buf[1], buf[2], buf[3]);
            let port = u16::from_be_bytes([buf[4], buf[5]]);
            Ok(Socks5Address::Ipv4(ip, port))
        }
        ADDR_TYPE_DOMAIN => {
            let mut len_buf = [0u8; 1];
            read_exact_with_timeout(stream, &mut len_buf).await?;
            let len = len_buf[0] as usize;
            let mut domain_buf = vec![0u8; len];
            read_exact_with_timeout(stream, &mut domain_buf).await?;
            let domain = String::from_utf8(domain_buf)
                .map_err(|e| PhantomError::ConnectionLost(format!("Invalid domain name: {}", e)))?;
            let mut port_buf = [0u8; 2];
            read_exact_with_timeout(stream, &mut port_buf).await?;
            let port = u16::from_be_bytes(port_buf);
            Ok(Socks5Address::Domain(domain, port))
        }
        ADDR_TYPE_IPV6 => {
            let mut buf = [0u8; 18];
            read_exact_with_timeout(stream, &mut buf).await?;
            let ip = Ipv6Addr::new(
                u16::from_be_bytes([buf[0], buf[1]]),
                u16::from_be_bytes([buf[2], buf[3]]),
                u16::from_be_bytes([buf[4], buf[5]]),
                u16::from_be_bytes([buf[6], buf[7]]),
                u16::from_be_bytes([buf[8], buf[9]]),
                u16::from_be_bytes([buf[10], buf[11]]),
                u16::from_be_bytes([buf[12], buf[13]]),
                u16::from_be_bytes([buf[14], buf[15]]),
            );
            let port = u16::from_be_bytes([buf[16], buf[17]]);
            Ok(Socks5Address::Ipv6(ip, port))
        }
        _ => Err(PhantomError::ConnectionLost(format!("Unsupported address type: {}", addr_type))),
    }
}

pub async fn send_reply(stream: &mut TcpStream, reply: u8, bound_addr: &Socks5Address) -> PhantomResult<()> {
    let mut response = Vec::with_capacity(10);
    response.push(SOCKS5_VERSION);
    response.push(reply);
    response.push(0x00);
    match bound_addr {
        Socks5Address::Ipv4(ip, port) => {
            response.push(ADDR_TYPE_IPV4);
            response.extend_from_slice(&ip.octets());
            response.extend_from_slice(&port.to_be_bytes());
        }
        Socks5Address::Ipv6(ip, port) => {
            response.push(ADDR_TYPE_IPV6);
            response.extend_from_slice(&ip.octets());
            response.extend_from_slice(&port.to_be_bytes());
        }
        Socks5Address::Domain(domain, port) => {
            let domain_bytes = domain.as_bytes();
            response.push(ADDR_TYPE_DOMAIN);
            response.push(domain_bytes.len() as u8);
            response.extend_from_slice(domain_bytes);
            response.extend_from_slice(&port.to_be_bytes());
        }
    }
    stream.write_all(&response).await.map_err(PhantomError::IoError)?;
    trace!("Sent SOCKS5 reply: reply={}", reply);
    Ok(())
}
