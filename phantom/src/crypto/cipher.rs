use ring::aead::{self, Aad, Nonce, NonceSequence, UnboundKey};
use ring::rand::{SecureRandom, SystemRandom};

use crate::utils::error::{PhantomError, PhantomResult};

pub struct OneNonceSequence(Option<Nonce>);

impl OneNonceSequence {
    pub fn new(nonce: Nonce) -> Self {
        Self(Some(nonce))
    }
}

impl NonceSequence for OneNonceSequence {
    fn advance(&mut self) -> Result<Nonce, ring::error::Unspecified> {
        self.0.take().ok_or(ring::error::Unspecified)
    }
}

pub struct PhantomCipher {
    sealing_key_bytes: [u8; 32],
    opening_key_bytes: [u8; 32],
}

impl PhantomCipher {
    pub fn new(sealing_key: [u8; 32], opening_key: [u8; 32]) -> PhantomResult<Self> {
        Ok(Self {
            sealing_key_bytes: sealing_key,
            opening_key_bytes: opening_key,
        })
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> PhantomResult<Vec<u8>> {
        let nonce = generate_nonce()?;
        let nonce_bytes = nonce.as_ref();

        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &self.sealing_key_bytes)
            .map_err(|e| PhantomError::CryptoError(format!("failed to create sealing key: {e}")))?;

        let mut sealing_key = aead::SealingKey::new(unbound_key, OneNonceSequence::new(nonce));

        let mut in_out = plaintext.to_vec();
        let tag = sealing_key
            .seal_in_place_separate_tag(Aad::empty(), &mut in_out)
            .map_err(|e| PhantomError::CryptoError(format!("encryption failed: {e}")))?;

        let mut ciphertext = Vec::with_capacity(nonce_bytes.len() + in_out.len() + tag.as_ref().len());
        ciphertext.extend_from_slice(nonce_bytes);
        ciphertext.append(&mut in_out);
        ciphertext.extend_from_slice(tag.as_ref());

        Ok(ciphertext)
    }

    pub fn decrypt(&self, ciphertext: &[u8]) -> PhantomResult<Vec<u8>> {
        if ciphertext.len() < Nonce::len() {
            return Err(PhantomError::CryptoError("ciphertext too short to contain nonce".to_string()));
        }

        let nonce_bytes = &ciphertext[..Nonce::len()];
        let encrypted_data = &ciphertext[Nonce::len()..];

        let nonce = Nonce::assume_unique_for_key(nonce_bytes);

        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &self.opening_key_bytes)
            .map_err(|e| PhantomError::CryptoError(format!("failed to create opening key: {e}")))?;

        let mut opening_key = aead::OpeningKey::new(unbound_key, OneNonceSequence::new(nonce));

        let mut in_out = encrypted_data.to_vec();
        let plaintext = opening_key
            .open_in_place(Aad::empty(), &mut in_out)
            .map_err(|e| PhantomError::CryptoError(format!("decryption failed: {e}")))?;

        Ok(plaintext.to_vec())
    }
}

fn generate_nonce() -> PhantomResult<Nonce> {
    let rng = SystemRandom::new();
    let mut nonce_bytes = [0u8; Nonce::len()];
    rng.fill(&mut nonce_bytes)
        .map_err(|e| PhantomError::CryptoError(format!("failed to generate nonce: {e}")))?;
    Ok(Nonce::assume_unique_for_key(nonce_bytes))
}
