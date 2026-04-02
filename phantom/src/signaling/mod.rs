pub mod doh;
pub mod nostr;
pub mod discovery;
pub mod failover;

pub use doh::DohDiscovery;
pub use nostr::NostrDiscovery;
pub use discovery::WorkerDiscovery;
pub use failover::FailoverManager;
