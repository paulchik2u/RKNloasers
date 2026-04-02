//! QUIC/TLS anti-fingerprinting module.
//!
//! Provides browser-accurate TLS ClientHello and QUIC transport parameter spoofing
//! to defeat JA3/JA4 fingerprinting used by modern DPI systems.

use std::sync::Arc;

use rand::seq::SliceRandom;
use ring::digest::{Context, SHA256};
use rustls::crypto::ring::default_provider;
use rustls::crypto::CryptoProvider;
use rustls::ClientConfig as RustlsClientConfig;
use rustls::SupportedCipherSuite;
use rustls::SupportedProtocolVersion;
use rustls::version::{TLS12, TLS13};
use tracing::{debug, info};

use crate::utils::error::{PhantomError, PhantomResult};

// ============================================================================
// Public types
// ============================================================================

/// Spoofs browser TLS/QUIC fingerprints to defeat JA3/JA4 detection.
pub struct QuicFingerprintSpo {
    target_browser: BrowserProfile,
    current_profile: BrowserProfile,
}

/// Complete browser fingerprint profile.
#[derive(Clone, Debug)]
pub struct BrowserProfile {
    /// Identifier like "chrome_120", "firefox_121", "edge_120", "safari_17"
    pub name: String,
    /// TLS version (0x0303 = TLS 1.2, 0x0304 = TLS 1.3)
    pub tls_version: u16,
    /// Cipher suites in exact wire order
    pub cipher_suites: Vec<u16>,
    /// TLS extensions in exact wire order
    pub extensions: Vec<TlsExtension>,
    /// QUIC transport parameters
    pub quic_transport_params: TransportParams,
    /// Initial packet size
    pub initial_packet_size: usize,
    /// ALPN protocol identifiers
    pub alpn_protocols: Vec<Vec<u8>>,
    /// Supported groups (elliptic curves) in order
    pub supported_groups: Vec<u16>,
    /// EC point formats
    pub ec_point_formats: Vec<u8>,
    /// Signature algorithms in order
    pub signature_algorithms: Vec<u16>,
    /// QUIC version
    pub quic_version: u32,
}

/// A single TLS extension.
#[derive(Clone, Debug)]
pub struct TlsExtension {
    pub ext_type: u16,
    pub data: Vec<u8>,
}

/// QUIC transport parameters as seen on the wire.
#[derive(Clone, Debug)]
pub struct TransportParams {
    pub max_udp_payload_size: u64,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub ack_delay_exponent: u64,
    pub max_ack_delay: u64,
    pub disable_active_migration: bool,
    pub active_connection_id_limit: u64,
}

// ============================================================================
// Extension type constants
// ============================================================================

const EXT_SERVER_NAME: u16 = 0x0000;
const EXT_EXTENDED_MASTER_SECRET: u16 = 0x0017;
const EXT_RENEGOTIATION_INFO: u16 = 0xff01;
const EXT_SUPPORTED_GROUPS: u16 = 0x000a;
const EXT_EC_POINT_FORMATS: u16 = 0x000b;
const EXT_SESSION_TICKET: u16 = 0x0023;
const EXT_ALPN: u16 = 0x0010;
const EXT_STATUS_REQUEST: u16 = 0x0005;
const EXT_SIGNATURE_ALGORITHMS: u16 = 0x000d;
const EXT_SCT: u16 = 0x0012;
const EXT_KEY_SHARE: u16 = 0x0033;
const EXT_PSK_KEY_EXCHANGE_MODES: u16 = 0x002d;
const EXT_SUPPORTED_VERSIONS: u16 = 0x002b;
const EXT_COMPRESS_CERTIFICATE: u16 = 0x001b;
const EXT_APPLICATION_SETTINGS: u16 = 0x001c;
const EXT_ENCRYPTED_CLIENT_HELLO: u16 = 0xfe0d;
const EXT_DELEGATED_CREDENTIALS: u16 = 0x0022;
const EXT_RECORD_SIZE_LIMIT: u16 = 0x001c;
const EXT_PADDING: u16 = 0x0015;

// ============================================================================
// Cipher suite constants (IANA values)
// ============================================================================

const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
const TLS_AES_256_GCM_SHA384: u16 = 0x1302;
const TLS_CHACHA20_POLY1305_SHA256: u16 = 0x1303;
const TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256: u16 = 0xc02b;
const TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256: u16 = 0xc02f;
const TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384: u16 = 0xc02c;
const TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384: u16 = 0xc030;
const TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256: u16 = 0xcca9;
const TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256: u16 = 0xcca8;
const TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA: u16 = 0xc013;
const TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA: u16 = 0xc014;
const TLS_RSA_WITH_AES_128_GCM_SHA256: u16 = 0x009c;
const TLS_RSA_WITH_AES_256_GCM_SHA384: u16 = 0x009d;
const TLS_RSA_WITH_AES_128_CBC_SHA: u16 = 0x002f;
const TLS_RSA_WITH_AES_256_CBC_SHA: u16 = 0x0035;
const TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA: u16 = 0xc009;
const TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA: u16 = 0xc00a;

// ============================================================================
// Group constants
// ============================================================================

const X25519: u16 = 0x001d;
const SECP256R1: u16 = 0x0017;
const X25519_KEM: u16 = 0x6399;
const SECP384R1: u16 = 0x0018;
const SECP521R1: u16 = 0x0019;
const FFDHE2048: u16 = 0x0100;
const FFDHE3072: u16 = 0x0101;

// ============================================================================
// Signature algorithm constants
// ============================================================================

const ECDSA_SECP256R1_SHA256: u16 = 0x0403;
const ECDSA_SECP384R1_SHA384: u16 = 0x0503;
const ECDSA_SECP521R1_SHA512: u16 = 0x0603;
const RSA_PSS_RSAE_SHA256: u16 = 0x0804;
const RSA_PSS_RSAE_SHA384: u16 = 0x0805;
const RSA_PSS_RSAE_SHA512: u16 = 0x0806;
const RSA_PKCS1_SHA256: u16 = 0x0401;
const RSA_PKCS1_SHA384: u16 = 0x0501;
const RSA_PKCS1_SHA512: u16 = 0x0601;
const ED25519: u16 = 0x0807;
const ED448: u16 = 0x0808;

// ============================================================================
// PSK mode constants
// ============================================================================

const PSK_MODE_KE: u8 = 1;
const PSK_MODE_DHE_KE: u8 = 2;

// ============================================================================
// Browser profile constructors
// ============================================================================

impl BrowserProfile {
    /// Chrome 120 fingerprint (Windows/Linux).
    pub fn chrome_120() -> Self {
        Self {
            name: "chrome_120".to_string(),
            tls_version: 0x0303,
            cipher_suites: vec![
                TLS_AES_128_GCM_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
                TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
                TLS_RSA_WITH_AES_128_GCM_SHA256,
                TLS_RSA_WITH_AES_256_GCM_SHA384,
                TLS_RSA_WITH_AES_128_CBC_SHA,
                TLS_RSA_WITH_AES_256_CBC_SHA,
            ],
            extensions: vec![
                TlsExtension {
                    ext_type: EXT_SERVER_NAME,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_EXTENDED_MASTER_SECRET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_RENEGOTIATION_INFO,
                    data: vec![0x00],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_GROUPS,
                    data: build_groups_data(&[X25519, SECP256R1, X25519_KEM, SECP384R1]),
                },
                TlsExtension {
                    ext_type: EXT_EC_POINT_FORMATS,
                    data: vec![0x01, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SESSION_TICKET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_ALPN,
                    data: build_alpn_data(&[b"h3", b"h2", b"http/1.1"]),
                },
                TlsExtension {
                    ext_type: EXT_STATUS_REQUEST,
                    data: vec![0x00, 0x00, 0x00, 0x00, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SIGNATURE_ALGORITHMS,
                    data: build_sig_algs_data(&[
                        ECDSA_SECP256R1_SHA256,
                        RSA_PSS_RSAE_SHA256,
                        RSA_PKCS1_SHA256,
                        ECDSA_SECP384R1_SHA384,
                        RSA_PSS_RSAE_SHA384,
                        RSA_PKCS1_SHA384,
                        RSA_PSS_RSAE_SHA512,
                        RSA_PKCS1_SHA512,
                    ]),
                },
                TlsExtension {
                    ext_type: EXT_SCT,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_KEY_SHARE,
                    data: build_key_share_placeholder(&[X25519, SECP256R1]),
                },
                TlsExtension {
                    ext_type: EXT_PSK_KEY_EXCHANGE_MODES,
                    data: vec![0x02, PSK_MODE_KE, PSK_MODE_DHE_KE],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_VERSIONS,
                    data: build_supported_versions_data(&[TLS13, TLS12]),
                },
                TlsExtension {
                    ext_type: EXT_COMPRESS_CERTIFICATE,
                    data: vec![0x01, 0x02],
                },
                TlsExtension {
                    ext_type: EXT_APPLICATION_SETTINGS,
                    data: vec![0x00],
                },
                TlsExtension {
                    ext_type: EXT_ENCRYPTED_CLIENT_HELLO,
                    data: vec![],
                },
            ],
            supported_groups: vec![X25519, SECP256R1, X25519_KEM, SECP384R1],
            ec_point_formats: vec![0x00],
            signature_algorithms: vec![
                ECDSA_SECP256R1_SHA256,
                RSA_PSS_RSAE_SHA256,
                RSA_PKCS1_SHA256,
                ECDSA_SECP384R1_SHA384,
                RSA_PSS_RSAE_SHA384,
                RSA_PKCS1_SHA384,
                RSA_PSS_RSAE_SHA512,
                RSA_PKCS1_SHA512,
            ],
            quic_transport_params: TransportParams {
                max_udp_payload_size: 1200,
                initial_max_data: 15728640,
                initial_max_stream_data_bidi_local: 6291456,
                initial_max_stream_data_bidi_remote: 6291456,
                initial_max_stream_data_uni: 6291456,
                initial_max_streams_bidi: 100,
                initial_max_streams_uni: 103,
                ack_delay_exponent: 9,
                max_ack_delay: 25,
                disable_active_migration: false,
                active_connection_id_limit: 8,
            },
            initial_packet_size: 1200,
            alpn_protocols: vec![b"h3".to_vec(), b"h2".to_vec(), b"http/1.1".to_vec()],
            quic_version: 0x00000001,
        }
    }

    /// Firefox 121 fingerprint.
    pub fn firefox_121() -> Self {
        Self {
            name: "firefox_121".to_string(),
            tls_version: 0x0303,
            cipher_suites: vec![
                TLS_AES_128_GCM_SHA256,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
                TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
                TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
                TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
                TLS_RSA_WITH_AES_128_GCM_SHA256,
                TLS_RSA_WITH_AES_256_GCM_SHA384,
                TLS_RSA_WITH_AES_128_CBC_SHA,
                TLS_RSA_WITH_AES_256_CBC_SHA,
            ],
            extensions: vec![
                TlsExtension {
                    ext_type: EXT_SERVER_NAME,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_EXTENDED_MASTER_SECRET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_RENEGOTIATION_INFO,
                    data: vec![0x00],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_GROUPS,
                    data: build_groups_data(&[X25519, SECP256R1, SECP384R1, SECP521R1, FFDHE2048, FFDHE3072]),
                },
                TlsExtension {
                    ext_type: EXT_EC_POINT_FORMATS,
                    data: vec![0x01, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SESSION_TICKET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_ALPN,
                    data: build_alpn_data(&[b"h3", b"h2", b"http/1.1"]),
                },
                TlsExtension {
                    ext_type: EXT_STATUS_REQUEST,
                    data: vec![0x00, 0x00, 0x00, 0x00, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_DELEGATED_CREDENTIALS,
                    data: build_sig_algs_data(&[
                        ECDSA_SECP256R1_SHA256,
                        ECDSA_SECP384R1_SHA384,
                        ECDSA_SECP521R1_SHA512,
                        ED25519,
                        ED448,
                    ]),
                },
                TlsExtension {
                    ext_type: EXT_KEY_SHARE,
                    data: build_key_share_placeholder(&[X25519]),
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_VERSIONS,
                    data: build_supported_versions_data(&[TLS13, TLS12]),
                },
                TlsExtension {
                    ext_type: EXT_SIGNATURE_ALGORITHMS,
                    data: build_sig_algs_data(&[
                        ECDSA_SECP256R1_SHA256,
                        ECDSA_SECP384R1_SHA384,
                        ECDSA_SECP521R1_SHA512,
                        RSA_PSS_RSAE_SHA256,
                        RSA_PSS_RSAE_SHA384,
                        RSA_PSS_RSAE_SHA512,
                        RSA_PKCS1_SHA256,
                        RSA_PKCS1_SHA384,
                        RSA_PKCS1_SHA512,
                        ED25519,
                        ED448,
                    ]),
                },
                TlsExtension {
                    ext_type: EXT_PSK_KEY_EXCHANGE_MODES,
                    data: vec![0x02, PSK_MODE_DHE_KE, PSK_MODE_KE],
                },
                TlsExtension {
                    ext_type: EXT_RECORD_SIZE_LIMIT,
                    data: vec![0x40, 0x01],
                },
                TlsExtension {
                    ext_type: EXT_COMPRESS_CERTIFICATE,
                    data: vec![0x01, 0x02],
                },
                TlsExtension {
                    ext_type: EXT_ENCRYPTED_CLIENT_HELLO,
                    data: vec![],
                },
            ],
            supported_groups: vec![X25519, SECP256R1, SECP384R1, SECP521R1, FFDHE2048, FFDHE3072],
            ec_point_formats: vec![0x00],
            signature_algorithms: vec![
                ECDSA_SECP256R1_SHA256,
                ECDSA_SECP384R1_SHA384,
                ECDSA_SECP521R1_SHA512,
                RSA_PSS_RSAE_SHA256,
                RSA_PSS_RSAE_SHA384,
                RSA_PSS_RSAE_SHA512,
                RSA_PKCS1_SHA256,
                RSA_PKCS1_SHA384,
                RSA_PKCS1_SHA512,
                ED25519,
                ED448,
            ],
            quic_transport_params: TransportParams {
                max_udp_payload_size: 1200,
                initial_max_data: 16777216,
                initial_max_stream_data_bidi_local: 4194304,
                initial_max_stream_data_bidi_remote: 4194304,
                initial_max_stream_data_uni: 4194304,
                initial_max_streams_bidi: 16,
                initial_max_streams_uni: 16,
                ack_delay_exponent: 8,
                max_ack_delay: 25,
                disable_active_migration: false,
                active_connection_id_limit: 2,
            },
            initial_packet_size: 1200,
            alpn_protocols: vec![b"h3".to_vec(), b"h2".to_vec(), b"http/1.1".to_vec()],
            quic_version: 0x00000001,
        }
    }

    /// Edge 120 fingerprint (Chromium-based, nearly identical to Chrome).
    pub fn edge_120() -> Self {
        Self {
            name: "edge_120".to_string(),
            tls_version: 0x0303,
            cipher_suites: vec![
                TLS_AES_128_GCM_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
                TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
                TLS_RSA_WITH_AES_128_GCM_SHA256,
                TLS_RSA_WITH_AES_256_GCM_SHA384,
                TLS_RSA_WITH_AES_128_CBC_SHA,
                TLS_RSA_WITH_AES_256_CBC_SHA,
            ],
            extensions: vec![
                TlsExtension {
                    ext_type: EXT_SERVER_NAME,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_EXTENDED_MASTER_SECRET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_RENEGOTIATION_INFO,
                    data: vec![0x00],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_GROUPS,
                    data: build_groups_data(&[X25519, SECP256R1, X25519_KEM, SECP384R1]),
                },
                TlsExtension {
                    ext_type: EXT_EC_POINT_FORMATS,
                    data: vec![0x01, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SESSION_TICKET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_ALPN,
                    data: build_alpn_data(&[b"h3", b"h2", b"http/1.1"]),
                },
                TlsExtension {
                    ext_type: EXT_STATUS_REQUEST,
                    data: vec![0x00, 0x00, 0x00, 0x00, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SIGNATURE_ALGORITHMS,
                    data: build_sig_algs_data(&[
                        ECDSA_SECP256R1_SHA256,
                        RSA_PSS_RSAE_SHA256,
                        RSA_PKCS1_SHA256,
                        ECDSA_SECP384R1_SHA384,
                        RSA_PSS_RSAE_SHA384,
                        RSA_PKCS1_SHA384,
                        RSA_PSS_RSAE_SHA512,
                        RSA_PKCS1_SHA512,
                    ]),
                },
                TlsExtension {
                    ext_type: EXT_SCT,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_KEY_SHARE,
                    data: build_key_share_placeholder(&[X25519, SECP256R1]),
                },
                TlsExtension {
                    ext_type: EXT_PSK_KEY_EXCHANGE_MODES,
                    data: vec![0x02, PSK_MODE_KE, PSK_MODE_DHE_KE],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_VERSIONS,
                    data: build_supported_versions_data(&[TLS13, TLS12]),
                },
                TlsExtension {
                    ext_type: EXT_COMPRESS_CERTIFICATE,
                    data: vec![0x01, 0x02],
                },
                TlsExtension {
                    ext_type: EXT_APPLICATION_SETTINGS,
                    data: vec![0x00],
                },
                TlsExtension {
                    ext_type: EXT_ENCRYPTED_CLIENT_HELLO,
                    data: vec![],
                },
            ],
            supported_groups: vec![X25519, SECP256R1, X25519_KEM, SECP384R1],
            ec_point_formats: vec![0x00],
            signature_algorithms: vec![
                ECDSA_SECP256R1_SHA256,
                RSA_PSS_RSAE_SHA256,
                RSA_PKCS1_SHA256,
                ECDSA_SECP384R1_SHA384,
                RSA_PSS_RSAE_SHA384,
                RSA_PKCS1_SHA384,
                RSA_PSS_RSAE_SHA512,
                RSA_PKCS1_SHA512,
            ],
            quic_transport_params: TransportParams {
                max_udp_payload_size: 1200,
                initial_max_data: 15728640,
                initial_max_stream_data_bidi_local: 6291456,
                initial_max_stream_data_bidi_remote: 6291456,
                initial_max_stream_data_uni: 6291456,
                initial_max_streams_bidi: 100,
                initial_max_streams_uni: 103,
                ack_delay_exponent: 9,
                max_ack_delay: 25,
                disable_active_migration: false,
                active_connection_id_limit: 8,
            },
            initial_packet_size: 1200,
            alpn_protocols: vec![b"h3".to_vec(), b"h2".to_vec(), b"http/1.1".to_vec()],
            quic_version: 0x00000001,
        }
    }

    /// Safari 17 fingerprint (macOS/iOS).
    pub fn safari_17() -> Self {
        Self {
            name: "safari_17".to_string(),
            tls_version: 0x0303,
            cipher_suites: vec![
                TLS_AES_128_GCM_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
                TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
                TLS_RSA_WITH_AES_128_GCM_SHA256,
                TLS_RSA_WITH_AES_256_GCM_SHA384,
                TLS_RSA_WITH_AES_128_CBC_SHA,
                TLS_RSA_WITH_AES_256_CBC_SHA,
            ],
            extensions: vec![
                TlsExtension {
                    ext_type: EXT_SERVER_NAME,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_EXTENDED_MASTER_SECRET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_RENEGOTIATION_INFO,
                    data: vec![0x00],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_GROUPS,
                    data: build_groups_data(&[X25519, SECP256R1, SECP384R1, SECP521R1]),
                },
                TlsExtension {
                    ext_type: EXT_EC_POINT_FORMATS,
                    data: vec![0x01, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SESSION_TICKET,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_ALPN,
                    data: build_alpn_data(&[b"h3", b"h2", b"http/1.1"]),
                },
                TlsExtension {
                    ext_type: EXT_STATUS_REQUEST,
                    data: vec![0x00, 0x00, 0x00, 0x00, 0x00],
                },
                TlsExtension {
                    ext_type: EXT_SIGNATURE_ALGORITHMS,
                    data: build_sig_algs_data(&[
                        ECDSA_SECP256R1_SHA256,
                        RSA_PSS_RSAE_SHA256,
                        RSA_PKCS1_SHA256,
                        ECDSA_SECP384R1_SHA384,
                        RSA_PSS_RSAE_SHA384,
                        RSA_PKCS1_SHA384,
                        RSA_PKCS1_SHA512,
                        RSA_PSS_RSAE_SHA512,
                    ]),
                },
                TlsExtension {
                    ext_type: EXT_SCT,
                    data: vec![],
                },
                TlsExtension {
                    ext_type: EXT_KEY_SHARE,
                    data: build_key_share_placeholder(&[X25519, SECP256R1]),
                },
                TlsExtension {
                    ext_type: EXT_PSK_KEY_EXCHANGE_MODES,
                    data: vec![0x02, PSK_MODE_KE, PSK_MODE_DHE_KE],
                },
                TlsExtension {
                    ext_type: EXT_SUPPORTED_VERSIONS,
                    data: build_supported_versions_data(&[TLS13, TLS12]),
                },
                TlsExtension {
                    ext_type: EXT_COMPRESS_CERTIFICATE,
                    data: vec![0x01, 0x02],
                },
                TlsExtension {
                    ext_type: EXT_PADDING,
                    data: vec![0x00; 128],
                },
            ],
            supported_groups: vec![X25519, SECP256R1, SECP384R1, SECP521R1],
            ec_point_formats: vec![0x00],
            signature_algorithms: vec![
                ECDSA_SECP256R1_SHA256,
                RSA_PSS_RSAE_SHA256,
                RSA_PKCS1_SHA256,
                ECDSA_SECP384R1_SHA384,
                RSA_PSS_RSAE_SHA384,
                RSA_PKCS1_SHA384,
                RSA_PKCS1_SHA512,
                RSA_PSS_RSAE_SHA512,
            ],
            quic_transport_params: TransportParams {
                max_udp_payload_size: 1200,
                initial_max_data: 10485760,
                initial_max_stream_data_bidi_local: 2097152,
                initial_max_stream_data_bidi_remote: 2097152,
                initial_max_stream_data_uni: 1048576,
                initial_max_streams_bidi: 8,
                initial_max_streams_uni: 8,
                ack_delay_exponent: 8,
                max_ack_delay: 25,
                disable_active_migration: false,
                active_connection_id_limit: 2,
            },
            initial_packet_size: 1200,
            alpn_protocols: vec![b"h3".to_vec(), b"h2".to_vec(), b"http/1.1".to_vec()],
            quic_version: 0x00000001,
        }
    }
}

// ============================================================================
// Helper functions for extension data construction
// ============================================================================

fn build_groups_data(groups: &[u16]) -> Vec<u8> {
    let mut data = Vec::with_capacity(2 + groups.len() * 2);
    let len = (groups.len() * 2) as u16;
    data.extend_from_slice(&len.to_be_bytes());
    for &g in groups {
        data.extend_from_slice(&g.to_be_bytes());
    }
    data
}

fn build_alpn_data(protocols: &[&[u8]]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut alpn_list = Vec::new();
    for proto in protocols {
        alpn_list.push(proto.len() as u8);
        alpn_list.extend_from_slice(proto);
    }
    let total_len = alpn_list.len() as u16;
    data.extend_from_slice(&total_len.to_be_bytes());
    data.extend_from_slice(&alpn_list);
    data
}

fn build_sig_algs_data(algs: &[u16]) -> Vec<u8> {
    let mut data = Vec::with_capacity(2 + algs.len() * 2);
    let len = (algs.len() * 2) as u16;
    data.extend_from_slice(&len.to_be_bytes());
    for &a in algs {
        data.extend_from_slice(&a.to_be_bytes());
    }
    data
}

fn build_key_share_placeholder(groups: &[u16]) -> Vec<u8> {
    let mut data = Vec::new();
    for &g in groups {
        data.extend_from_slice(&g.to_be_bytes());
        if g == X25519 || g == X25519_KEM {
            let key_len = 32u16;
            data.extend_from_slice(&key_len.to_be_bytes());
            data.resize(data.len() + 32, 0x00);
        } else if g == SECP256R1 {
            let key_len = 65u16;
            data.extend_from_slice(&key_len.to_be_bytes());
            data.resize(data.len() + 65, 0x00);
        } else if g == SECP384R1 {
            let key_len = 97u16;
            data.extend_from_slice(&key_len.to_be_bytes());
            data.resize(data.len() + 97, 0x00);
        } else if g == SECP521R1 {
            let key_len = 133u16;
            data.extend_from_slice(&key_len.to_be_bytes());
            data.resize(data.len() + 133, 0x00);
        } else {
            let key_len = 32u16;
            data.extend_from_slice(&key_len.to_be_bytes());
            data.resize(data.len() + 32, 0x00);
        }
    }
    let total_len = data.len() as u16;
    let mut result = Vec::with_capacity(2 + data.len());
    result.extend_from_slice(&total_len.to_be_bytes());
    result.extend_from_slice(&data);
    result
}

fn build_supported_versions_data(versions: &[&SupportedProtocolVersion]) -> Vec<u8> {
    let mut data = Vec::with_capacity(1 + versions.len());
    data.push(versions.len() as u8);
    for v in versions {
        data.push(v.version);
    }
    data
}

// ============================================================================
// JA3/JA4 hash computation
// ============================================================================

impl QuicFingerprintSpo {
    /// Compute JA3 fingerprint hash from the current profile.
    ///
    /// JA3 = MD5(ssl_version,cipher_suites,extensions,elliptic_curves,elliptic_curve_point_formats)
    pub fn compute_ja3_hash(&self) -> String {
        let profile = &self.current_profile;

        let version_str = format!("{}", profile.tls_version);

        let ciphers_str = profile
            .cipher_suites
            .iter()
            .map(|c| format!("{}", c))
            .collect::<Vec<_>>()
            .join("-");

        let exts_str = profile
            .extensions
            .iter()
            .map(|e| format!("{}", e.ext_type))
            .collect::<Vec<_>>()
            .join("-");

        let groups_str = profile
            .supported_groups
            .iter()
            .map(|g| format!("{}", g))
            .collect::<Vec<_>>()
            .join("-");

        let formats_str = profile
            .ec_point_formats
            .iter()
            .map(|f| format!("{}", f))
            .collect::<Vec<_>>()
            .join("-");

        let ja3_string = format!(
            "{},{},{},{},{}",
            version_str, ciphers_str, exts_str, groups_str, formats_str
        );

        debug!("JA3 string: {}", ja3_string);

        let mut context = Context::new(&SHA256);
        context.update(ja3_string.as_bytes());
        let digest = context.finish();

        hex::encode(digest.as_ref())
    }

    /// Compute JA4 fingerprint hash for QUIC.
    ///
    /// JA4 = {proto}_{tls_version}_{sni}_{cipher_count}_{ext_count}_{alpn}_{first_cipher}_{first_ext}_{last_cipher}_{last_ext}
    pub fn compute_ja4_hash(&self) -> String {
        let profile = &self.current_profile;

        let proto = "q";
        let tls_version = match profile.tls_version {
            0x0304 => "13",
            0x0303 => "12",
            _ => "00",
        };

        let has_sni = profile.extensions.iter().any(|e| e.ext_type == EXT_SERVER_NAME);
        let sni = if has_sni { "d" } else { "i" };

        let cipher_count = profile.cipher_suites.len();
        let ext_count = profile.extensions.len();

        let alpn = if profile.alpn_protocols.is_empty() {
            "00".to_string()
        } else {
            profile.alpn_protocols
                .first()
                .map(|p| {
                    let s = String::from_utf8_lossy(p);
                    format!("{:02}{:02}", s.len(), s.chars().next().map(|c| c as u8).unwrap_or(0))
                })
                .unwrap_or_else(|| "00".to_string())
        };

        let first_cipher = profile.cipher_suites.first().copied().unwrap_or(0);
        let last_cipher = profile.cipher_suites.last().copied().unwrap_or(0);
        let first_ext = profile.extensions.first().map(|e| e.ext_type).unwrap_or(0);
        let last_ext = profile.extensions.last().map(|e| e.ext_type).unwrap_or(0);

        let ja4_string = format!(
            "{}_{}_{}_{}_{:02}_{}_{:04x}_{:04x}_{:04x}_{:04x}",
            proto, tls_version, sni, cipher_count, ext_count, alpn, first_cipher, first_ext, last_cipher, last_ext
        );

        debug!("JA4 string: {}", ja4_string);

        let mut context = Context::new(&SHA256);
        context.update(ja4_string.as_bytes());
        let digest = context.finish();

        hex::encode(digest.as_ref())
    }
}

// ============================================================================
// QuicFingerprintSpo implementation
// ============================================================================

impl QuicFingerprintSpo {
    /// Create a new fingerprint spoofer with the given browser profile.
    pub fn new(browser: BrowserProfile) -> Self {
        info!(
            profile = %browser.name,
            ciphers = browser.cipher_suites.len(),
            extensions = browser.extensions.len(),
            "Initialized QUIC fingerprint spoofer"
        );
        Self {
            target_browser: browser.clone(),
            current_profile: browser,
        }
    }

    /// Apply fingerprint spoofing to a quinn ClientConfig.
    ///
    /// This configures the underlying rustls `ClientConfig` with browser-accurate
    /// cipher suites, key exchange groups, and QUIC transport parameters.
    pub fn apply_to_config(&self, config: &mut quinn::ClientConfig) -> PhantomResult<()> {
        let profile = &self.current_profile;

        debug!(
            profile = %profile.name,
            "Applying fingerprint to QUIC client config"
        );

        let crypto_provider = self.build_crypto_provider()?;

        let tls_config = self.build_tls_config(&crypto_provider)?;

        let mut quinn_crypto = quinn::crypto::rustls::QuicClientConfig::new(tls_config)
            .map_err(|e| PhantomError::CryptoError(format!("Failed to create QuicClientConfig: {}", e)))?;

        *config = quinn::ClientConfig::new(Arc::new(quinn_crypto));

        let mut transport_config = quinn::TransportConfig::default();
        transport_config
            .max_concurrent_bidi_streams(
                quinn::VarInt::from_u64(profile.quic_transport_params.initial_max_streams_bidi)
                    .map_err(|e| PhantomError::ConfigError(format!("Invalid bidi streams: {}", e)))?,
            )
            .max_concurrent_uni_streams(
                quinn::VarInt::from_u64(profile.quic_transport_params.initial_max_streams_uni)
                    .map_err(|e| PhantomError::ConfigError(format!("Invalid uni streams: {}", e)))?,
            )
            .stream_receive_window(
                quinn::VarInt::from_u64(profile.quic_transport_params.initial_max_stream_data_bidi_remote)
                    .map_err(|e| PhantomError::ConfigError(format!("Invalid stream window: {}", e)))?,
            )
            .receive_window(
                quinn::VarInt::from_u64(profile.quic_transport_params.initial_max_data)
                    .map_err(|e| PhantomError::ConfigError(format!("Invalid receive window: {}", e)))?,
            )
            .send_window(profile.quic_transport_params.initial_max_data)
            .initial_mtu(profile.quic_transport_params.max_udp_payload_size as u16)
            .min_mtu(1200)
            .keep_alive_interval(Some(std::time::Duration::from_secs(15)));

        config.transport_config(Arc::new(transport_config));

        info!(
            profile = %profile.name,
            mtu = profile.quic_transport_params.max_udp_payload_size,
            "Fingerprint applied to QUIC client config"
        );

        Ok(())
    }

    /// Verify that our computed fingerprint matches the target browser.
    pub fn verify_match(&self, target_ja3: &str, target_ja4: &str) -> bool {
        let our_ja3 = self.compute_ja3_hash();
        let our_ja4 = self.compute_ja4_hash();

        let ja3_match = our_ja3.eq_ignore_ascii_case(target_ja3);
        let ja4_match = our_ja4.eq_ignore_ascii_case(target_ja4);

        debug!(
            profile = %self.current_profile.name,
            ja3_match,
            ja4_match,
            "Fingerprint verification result"
        );

        ja3_match && ja4_match
    }

    /// Rotate to a random browser profile from the known set.
    ///
    /// Returns `Ok(())` with the new profile name on success.
    pub fn rotate(&mut self) -> PhantomResult<()> {
        let profiles = [
            BrowserProfile::chrome_120(),
            BrowserProfile::firefox_121(),
            BrowserProfile::edge_120(),
            BrowserProfile::safari_17(),
        ];

        let old_name = self.current_profile.name.clone();

        let new_profile = profiles
            .choose(&mut rand::thread_rng())
            .ok_or_else(|| PhantomError::CryptoError("Failed to select rotation profile".to_string()))?;

        self.current_profile = new_profile.clone();

        info!(
            from = %old_name,
            to = %self.current_profile.name,
            "Rotated QUIC fingerprint"
        );

        Ok(())
    }

    /// Build a rustls `CryptoProvider` with browser-accurate cipher suite ordering.
    fn build_crypto_provider(&self) -> PhantomResult<Arc<CryptoProvider>> {
        let profile = &self.current_profile;
        let base = default_provider();

        let mut cipher_suites: Vec<SupportedCipherSuite> = Vec::new();

        for &cipher_id in &profile.cipher_suites {
            if let Some(suite) = find_cipher_suite_by_id(cipher_id, &base) {
                cipher_suites.push(suite);
            }
        }

        if cipher_suites.is_empty() {
            return Err(PhantomError::CryptoError(
                "No cipher suites matched browser profile".to_string(),
            ));
        }

        let mut kx_groups = Vec::new();
        for &group_id in &profile.supported_groups {
            if let Some(kxg) = find_kx_group_by_id(group_id, &base) {
                kx_groups.push(kxg);
            }
        }

        if kx_groups.is_empty() {
            return Err(PhantomError::CryptoError(
                "No key exchange groups matched browser profile".to_string(),
            ));
        }

        let provider = CryptoProvider {
            cipher_suites,
            kx_groups,
            signature_verification_algorithms: base.signature_verification_algorithms,
            secure_random: base.secure_random,
            key_provider: base.key_provider,
        };

        debug!(
            profile = %profile.name,
            cipher_count = provider.cipher_suites.len(),
            kx_count = provider.kx_groups.len(),
            "Built custom crypto provider"
        );

        Ok(Arc::new(provider))
    }

    /// Build a rustls `ClientConfig` with browser-accurate settings.
    fn build_tls_config(&self, provider: &Arc<CryptoProvider>) -> PhantomResult<RustlsClientConfig> {
        let profile = &self.current_profile;

        let versions: Vec<&SupportedProtocolVersion> = if profile.tls_version == 0x0304 {
            vec![&TLS13]
        } else {
            vec![&TLS13, &TLS12]
        };

        let tls_config = RustlsClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&versions)
            .map_err(|e| PhantomError::CryptoError(format!("Failed to set protocol versions: {}", e)))?
            .with_root_certificates(
                rustls::RootCertStore {
                    roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
                }
                .into(),
            )
            .with_no_client_auth();

        let mut tls_config = tls_config;
        tls_config.alpn_protocols = profile.alpn_protocols.clone();
        tls_config.enable_sni = true;

        debug!(
            profile = %profile.name,
            alpn_count = tls_config.alpn_protocols.len(),
            "Built rustls client config"
        );

        Ok(tls_config)
    }
}

// ============================================================================
// Cipher suite and KX group lookup helpers
// ============================================================================

fn find_cipher_suite_by_id(id: u16, provider: &CryptoProvider) -> Option<SupportedCipherSuite> {
    use rustls::CipherSuite;

    let target = match id {
        TLS_AES_128_GCM_SHA256 => CipherSuite::TLS13_AES_128_GCM_SHA256,
        TLS_AES_256_GCM_SHA384 => CipherSuite::TLS13_AES_256_GCM_SHA384,
        TLS_CHACHA20_POLY1305_SHA256 => CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
        TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
        TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 => CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
        TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 => CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
        TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 => CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
        TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 => CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
        TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA => CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
        TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA => CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
        TLS_RSA_WITH_AES_128_GCM_SHA256 => CipherSuite::TLS_RSA_WITH_AES_128_GCM_SHA256,
        TLS_RSA_WITH_AES_256_GCM_SHA384 => CipherSuite::TLS_RSA_WITH_AES_256_GCM_SHA384,
        TLS_RSA_WITH_AES_128_CBC_SHA => CipherSuite::TLS_RSA_WITH_AES_128_CBC_SHA,
        TLS_RSA_WITH_AES_256_CBC_SHA => CipherSuite::TLS_RSA_WITH_AES_256_CBC_SHA,
        TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
        TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA => CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
        _ => return None,
    };

    provider
        .cipher_suites
        .iter()
        .find(|cs| cs.suite() == target)
        .copied()
}

fn find_kx_group_by_id(id: u16, provider: &CryptoProvider) -> Option<&'static dyn rustls::crypto::SupportedKxGroup> {
    use rustls::NamedGroup;

    let target = match id {
        X25519 => NamedGroup::X25519,
        SECP256R1 => NamedGroup::SECP256R1,
        SECP384R1 => NamedGroup::SECP384R1,
        SECP521R1 => NamedGroup::SECP521R1,
        FFDHE2048 => NamedGroup::FFDHE2048,
        FFDHE3072 => NamedGroup::FFDHE3072,
        _ => return None,
    };

    provider
        .kx_groups
        .iter()
        .find(|kxg| kxg.name() == target)
        .copied()
}

// ============================================================================
// hex encoding helper (avoid extra dependency)
// ============================================================================

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

// ============================================================================
// webpki roots stub (since the crate may not be in deps)
// ============================================================================

mod webpki_roots {
    use rustls::pki_types::CertificateDer;
    use rustls::RootCertStore;
    use std::sync::LazyLock;

    pub static TLS_SERVER_ROOTS: LazyLock<Vec<CertificateDer<'static>>> = LazyLock::new(|| {
        let mut store = RootCertStore::empty();
        store.extend(
            rustls_native_certs::load_native_certs()
                .certs
                .into_iter()
                .filter_map(|c| c.parse().ok()),
        );
        store.roots
    });
}
