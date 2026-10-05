//! Forge access-token storage outside the database. Tokens live in the OS keychain (Secret
//! Service on Linux, Keychain on macOS, Credential Manager on Windows), keyed by forge id, so a
//! DB file carries no secret.
//! `CredentialStore` is a trait so an in-memory fake can make the host logic unit-testable
//! headless.

use std::collections::HashMap;
use std::sync::Mutex;

/// Stores/loads a forge's access token, keyed by forge id. Sync (keychain ops are fast).
pub trait CredentialStore: Send + Sync {
    fn get(&self, forge_id: &str) -> Option<String>;
    fn set(&self, forge_id: &str, token: &str) -> Result<(), String>;
    /// Remove a forge's token. Removing an absent one is a no-op.
    fn delete(&self, forge_id: &str) -> Result<(), String>;
    /// Whether a credential exists for the forge (the UI's "connected" status).
    fn has(&self, forge_id: &str) -> bool {
        self.get(forge_id).is_some()
    }
}

/// The OS keychain via the `keyring` crate. One entry per forge under the `orgonzola` service.
pub struct KeyringCredentials {
    service: String,
}

impl KeyringCredentials {
    pub fn new() -> Self {
        Self {
            service: "orgonzola".to_string(),
        }
    }

    fn entry(&self, forge_id: &str) -> Result<keyring::Entry, String> {
        keyring::Entry::new(&self.service, forge_id).map_err(|e| e.to_string())
    }
}

impl Default for KeyringCredentials {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialStore for KeyringCredentials {
    fn get(&self, forge_id: &str) -> Option<String> {
        match self.entry(forge_id).ok()?.get_password() {
            Ok(token) => Some(token),
            Err(keyring::Error::NoEntry) => None,
            Err(e) => {
                eprintln!("keychain read failed for {forge_id}: {e}");
                None
            }
        }
    }

    fn set(&self, forge_id: &str, token: &str) -> Result<(), String> {
        self.entry(forge_id)?
            .set_password(token)
            .map_err(|e| e.to_string())
    }

    fn delete(&self, forge_id: &str) -> Result<(), String> {
        match self.entry(forge_id)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// An in-memory credential store for tests and as a fallback when no keychain is available.
#[derive(Default)]
pub struct MemoryCredentials {
    map: Mutex<HashMap<String, String>>,
}

impl CredentialStore for MemoryCredentials {
    fn get(&self, forge_id: &str) -> Option<String> {
        self.map.lock().ok()?.get(forge_id).cloned()
    }

    fn set(&self, forge_id: &str, token: &str) -> Result<(), String> {
        self.map
            .lock()
            .map_err(|e| e.to_string())?
            .insert(forge_id.to_string(), token.to_string());
        Ok(())
    }

    fn delete(&self, forge_id: &str) -> Result<(), String> {
        self.map.lock().map_err(|e| e.to_string())?.remove(forge_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_round_trips_and_delete_is_idempotent() {
        let c = MemoryCredentials::default();
        assert!(c.get("gh").is_none());
        assert!(!c.has("gh"));
        c.set("gh", "secret").unwrap();
        assert_eq!(c.get("gh").as_deref(), Some("secret"));
        assert!(c.has("gh"));
        c.set("gh", "rotated").unwrap();
        assert_eq!(c.get("gh").as_deref(), Some("rotated"));
        c.delete("gh").unwrap();
        assert!(!c.has("gh"));
        c.delete("gh").unwrap(); // removing an absent one is a no-op
    }
}
