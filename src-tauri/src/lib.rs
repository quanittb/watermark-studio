mod commands;
mod error;
mod media;
mod project;
mod tracking;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(commands::project::AppState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::project::open_video,
            commands::project::get_project,
            commands::project::extract_preview_frame,
            commands::project::save_watermark_anchor,
            commands::project::analyze_track,
            commands::project::retrack_track,
            commands::project::cancel_tracking,
            commands::project::interpolate_tracking_range,
            commands::project::save_manual_anchor,
            commands::project::accept_tracking_frame,
            commands::project::save_removal_config,
            commands::project::render_video,
            commands::project::cancel_render,
            commands::project::get_project_asset_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
