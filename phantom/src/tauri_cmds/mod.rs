use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use tauri::State;

use crate::core::engine::PhantomEngine;
use crate::core::state::ConnectionState;

#[derive(Serialize, Deserialize)]
pub struct StatusResponse {
    pub state: String,
    pub speed_up: u64,
    pub speed_down: u64,
    pub worker_count: usize,
    pub error_message: Option<String>,
}

#[tauri::command]
pub async fn cmd_connect(
    engine: State<'_, Arc<Mutex<PhantomEngine>>>,
) -> Result<String, String> {
    let mut engine = engine.lock().await;
    engine.start().await.map_err(|e| e.to_string())?;
    Ok("Connected".to_string())
}

#[tauri::command]
pub async fn cmd_disconnect(
    engine: State<'_, Arc<Mutex<PhantomEngine>>>,
) -> Result<String, String> {
    let mut engine = engine.lock().await;
    engine.stop().await.map_err(|e| e.to_string())?;
    Ok("Disconnected".to_string())
}

#[tauri::command]
pub async fn cmd_get_status(
    engine: State<'_, Arc<Mutex<PhantomEngine>>>,
) -> Result<StatusResponse, String> {
    let engine = engine.lock().await;
    let state = engine.get_state().await;

    match state {
        ConnectionState::Disconnected => Ok(StatusResponse {
            state: "disconnected".to_string(),
            speed_up: 0,
            speed_down: 0,
            worker_count: 0,
            error_message: None,
        }),
        ConnectionState::Connecting => Ok(StatusResponse {
            state: "connecting".to_string(),
            speed_up: 0,
            speed_down: 0,
            worker_count: 0,
            error_message: None,
        }),
        ConnectionState::Connected {
            speed_up,
            speed_down,
            worker_count,
        } => Ok(StatusResponse {
            state: "connected".to_string(),
            speed_up,
            speed_down,
            worker_count,
            error_message: None,
        }),
        ConnectionState::Error { message } => Ok(StatusResponse {
            state: "error".to_string(),
            speed_up: 0,
            speed_down: 0,
            worker_count: 0,
            error_message: Some(message),
        }),
    }
}

#[tauri::command]
pub async fn cmd_get_config(
    engine: State<'_, Arc<Mutex<PhantomEngine>>>,
) -> Result<String, String> {
    let engine = engine.lock().await;
    serde_json::to_string(&engine.config).map_err(|e| e.to_string())
}
