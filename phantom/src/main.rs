use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    tracing::info!("Phantom starting");

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config/phantom.json".to_string());

    let config = phantom::phantom_core::config::PhantomConfig::load_from_file(&config_path)
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to load config from {}: {}, using defaults", config_path, e);
            phantom::phantom_core::config::PhantomConfig::default()
        });

    tracing::info!("Configuration loaded: protocol={}", config.transport.protocol);

    let mut engine = phantom::phantom_core::engine::PhantomEngine::new(config);
    engine.start().await?;

    tracing::info!("Phantom engine started");

    let handle = tauri::Builder::default()
        .manage(std::sync::Arc::new(tokio::sync::Mutex::new(engine)))
        .invoke_handler(tauri::generate_handler![
            phantom::tauri_cmds::connect,
            phantom::tauri_cmds::disconnect,
            phantom::tauri_cmds::get_status,
            phantom::tauri_cmds::get_config,
        ])
        .build(tauri::generate_context!())
        .expect("Failed to build Tauri application");

    let app = tauri::Builder::default()
        .manage(std::sync::Arc::new(tokio::sync::Mutex::new(engine)))
        .invoke_handler(tauri::generate_handler![
            phantom::tauri_cmds::cmd_connect,
            phantom::tauri_cmds::cmd_disconnect,
            phantom::tauri_cmds::cmd_get_status,
            phantom::tauri_cmds::cmd_get_config,
        ])
        .setup(|app| {
            tracing::info!("Phantom Tauri app setup complete");
            let _handle = app.handle();
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Failed to build Tauri application");

    app.run(|_app_handle, _event| {});

    tracing::info!("Phantom shutting down");
    Ok(())
}
