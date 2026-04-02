use quinn::{ClientConfig, Endpoint};
use std::sync::Arc;

use crate::utils::{PhantomError, PhantomResult};

pub struct QuicEndpoint {
    endpoint: Endpoint,
}

impl QuicEndpoint {
    pub fn new() -> PhantomResult<Self> {
        let client_config = Self::build_client_config()?;

        let runtime = quinn::default_runtime().ok_or_else(|| {
            PhantomError::ConnectionFailed("Failed to obtain async runtime".to_string())
        })?;

        let endpoint = Endpoint::client("0.0.0.0:0".parse().map_err(|e| {
            PhantomError::ConnectionFailed(format!("Invalid bind address: {}", e))
        })?)
        .map_err(|e| PhantomError::ConnectionFailed(format!("Failed to bind endpoint: {}", e)))?;

        runtime.spawn(async move {
            let _ = endpoint.wait_idle().await;
        });

        endpoint.set_default_client_config(client_config);

        tracing::info!("QUIC endpoint bound to 0.0.0.0:0");

        Ok(Self { endpoint })
    }

    pub fn build_client_config() -> PhantomResult<ClientConfig> {
        let mut root_store = rustls::RootCertStore::empty();

        let certs = rustls_native_certs::load_native_certs();

        if !certs.errors.is_empty() {
            tracing::warn!(
                errors = ?certs.errors,
                "Some certificates failed to load from native store"
            );
        }

        for cert in certs.certs {
            root_store
                .add(cert)
                .map_err(|e| PhantomError::CryptoError(format!("Failed to add root cert: {}", e)))?;
        }

        tracing::info!(certs_loaded = root_store.len(), "Loaded root certificates");

        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let quic_config = quinn::crypto::rustls::QuicClientConfig::try_from(tls_config)
            .map_err(|e| PhantomError::CryptoError(format!("Failed to build QUIC TLS config: {}", e)))?;

        let mut client_config = ClientConfig::new(Arc::new(quic_config));

        let mut transport_config = quinn::TransportConfig::default();
        transport_config.keep_alive_interval(Some(std::time::Duration::from_secs(4)));
        transport_config.max_idle_timeout(Some(std::time::Duration::from_secs(10).try_into().map_err(
            |e| PhantomError::ConfigError(format!("Invalid idle timeout: {}", e)),
        )?));

        client_config.transport_config(Arc::new(transport_config));

        Ok(client_config)
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }
}
