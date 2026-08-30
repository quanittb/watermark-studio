mod commands;
mod error;
mod jobs;
mod media;
mod project;
mod tracking;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(commands::project::AppState::default())
        .setup(|app| {
            let state = app.state::<commands::project::AppState>();
            commands::project::start_pending_job_worker(app.handle().clone(), state);
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            commands::project::open_video,
            commands::project::get_project,
            commands::project::list_projects,
            commands::project::remove_project,
            commands::project::extract_preview_frame,
            commands::project::extract_focus_preview,
            commands::project::save_watermark_anchor,
            commands::project::create_calibration_profile,
            commands::project::auto_calibrate_best_quality,
            commands::project::save_calibration_mask_edit,
            commands::project::analyze_track,
            commands::project::retrack_track,
            commands::project::cancel_tracking,
            commands::project::interpolate_tracking_range,
            commands::project::save_manual_anchor,
            commands::project::accept_tracking_frame,
            commands::project::mark_occluded_range,
            commands::project::save_removal_config,
            commands::project::render_video,
            commands::project::render_best_quality_video,
            commands::project::list_jobs,
            commands::project::enqueue_best_quality_job,
            commands::project::cancel_job,
            commands::project::regen_job,
            commands::project::suggest_best_quality_samples,
            commands::project::detect_hardware,
            commands::project::cancel_render,
            commands::project::get_project_asset_path,
            commands::project::read_project_asset_bytes
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
