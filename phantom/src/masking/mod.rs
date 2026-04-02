pub mod adversarial;
pub mod burst_engine;
pub mod dpi_simulator;
pub mod kcp_wrapper;
pub mod timing;
pub mod timing_simulator;
pub mod packet;
pub mod profiles;
pub mod protocol_blender;
pub mod statistical_model;

pub use adversarial::{AdversarialMasker, MaskingDecision, PacketDirection, PacketRecord, WindowStats};
pub use burst_engine::{BurstEngine, BurstState, BurstType, PacketAction};
pub use dpi_simulator::{
    ClassificationResult, DpiAnalysis, DpiRegion, DpiSignature, DpiSimulator, DirectionAnalysis,
    EntropyAnalysis, PeriodicityAnalysis, RiskLevel, SizeAnalysis, TimingAnalysis, TrafficClassifier,
    TrafficProfile, TrafficType,
};
pub use kcp_wrapper::KcpWrapper;
pub use timing::TrafficShaper;
pub use timing_simulator::{
    AdaptiveTimingController, Oscillator, PacketTimingSimulator, PulseGenerator,
    TimingStatistics, TimingValidation,
};
pub use packet::PacketBuilder;
pub use profiles::GamingProfile;
pub use protocol_blender::{
    BlendedPacket, DirectionPattern, EncryptionType, ProtocolBlender, ProtocolProfile,
    ProtocolSignature, ProtocolStatistics, create_default_profiles,
};
pub use statistical_model::{
    BurstPattern, EntropyProfile, GaussianComponent, PacketSizeDistribution,
    StatisticalGameTrafficModel, TimingDistribution,
};
