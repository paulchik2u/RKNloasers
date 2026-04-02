use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use ring::rand::{SecureRandom, SystemRandom};

use crate::crypto::cipher::PhantomCipher;
use crate::transport::QuicClient;
use crate::utils::error::{PhantomError, PhantomResult};

pub struct KeyRotator {
    cipher: Arc<RwLock<PhantomCipher>>,
    rotation_interval: Duration,
    last_rotation: RwLock<Instant>,
    exit_pubkey: Option<[u8; 32]>,
}

impl KeyRotator {
    pub fn new(rotation_minutes: u64) -> Self {
        let sealing_key = generate_random_key();
        let opening_key = generate_random_key();
        let cipher = PhantomCipher::new(sealing_key, opening_key)
            .unwrap_or_else(|_| PhantomCipher::new([0u8; 32], [0u8; 32]).unwrap());

        Self {
            cipher: Arc::new(RwLock::new(cipher)),
            rotation_interval: Duration::from_secs(rotation_minutes * 60),
            last_rotation: RwLock::new(Instant::now()),
            exit_pubkey: None,
        }
    }

    pub fn should_rotate(&self) -> bool {
        self.last_rotation
            .read()
            .map(|t| t.elapsed() >= self.rotation_interval)
            .unwrap_or(false)
    }

    pub async fn rotate(&self, quic_client: &QuicClient) -> PhantomResult<()> {
        let (new_sealing, new_opening) = Self::generate_keypair();

        let pubkey_bytes = new_sealing.to_vec();
        quic_client
            .send_key_update(&pubkey_bytes)
            .await
            .map_err(|e| PhantomError::CryptoError(format!("key rotation send failed: {e}")))?;

        let new_cipher = PhantomCipher::new(new_sealing, new_opening)?;

        {
            let mut cipher_guard = self.cipher.write().map_err(|e| {
                PhantomError::CryptoError(format!("failed to acquire cipher write lock: {e}"))
            })?;
            *cipher_guard = new_cipher;
        }

        {
            let mut last = self.last_rotation.write().map_err(|e| {
                PhantomError::CryptoError(format!("failed to acquire last_rotation write lock: {e}"))
            })?;
            *last = Instant::now();
        }

        Ok(())
    }

    pub fn generate_keypair() -> ([u8; 32], [u8; 32]) {
        (generate_random_key(), generate_random_key())
    }

    pub fn get_cipher(&self) -> Arc<RwLock<PhantomCipher>> {
        Arc::clone(&self.cipher)
    }
}

fn generate_random_key() -> [u8; 32] {
    let rng = SystemRandom::new();
    let mut key = [0u8; 32];
    rng.fill(&mut key).unwrap_or(key);
    key
}
