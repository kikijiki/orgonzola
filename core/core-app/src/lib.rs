//! Core application logic: the health probe the shell answers its status command with.
//! Host-agnostic: no Tauri, no display. The shell wires this into a typed command.

use core_model::Health;

/// The core crate version, surfaced to the UI through the health command.
pub fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Answer the health probe. Pure and synchronous: usable with the LLM disabled and with no host.
pub fn health() -> Health {
    Health {
        status: "ok".to_string(),
        core_version: core_version().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_is_ok() {
        let h = health();
        assert_eq!(h.status, "ok");
        assert_eq!(h.core_version, core_version());
    }
}
