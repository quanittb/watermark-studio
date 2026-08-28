use crate::error::AppError;
use crate::project::model::Project;
use std::fs;
use std::path::{Path, PathBuf};

const PROJECTS_DIRECTORY: &str = "projects";
const PROJECT_BACKUP_FILE: &str = "project.json.last-good.bak";

pub fn projects_root(app_data_dir: &Path) -> PathBuf {
    app_data_dir
        .join("WatermarkStudio")
        .join(PROJECTS_DIRECTORY)
}

pub fn project_directory(app_data_dir: &Path, project_id: &str) -> Result<PathBuf, AppError> {
    if project_id.is_empty()
        || project_id.contains(['/', '\\'])
        || project_id == "."
        || project_id == ".."
    {
        return Err(AppError::InvalidRequest("Invalid project id.".to_string()));
    }

    Ok(projects_root(app_data_dir).join(project_id))
}

pub fn create_project_workspace(
    app_data_dir: &Path,
    project: &Project,
) -> Result<PathBuf, AppError> {
    let directory = project_directory(app_data_dir, &project.id)?;
    fs::create_dir_all(directory.join("frames"))?;
    fs::create_dir_all(directory.join("templates"))?;
    fs::create_dir_all(directory.join("cache"))?;
    save_project_atomic(&directory, project)?;
    Ok(directory)
}

pub fn load_project(app_data_dir: &Path, project_id: &str) -> Result<Project, AppError> {
    let directory = project_directory(app_data_dir, project_id)?;
    let project_path = directory.join("project.json");
    let backup_path = directory.join(PROJECT_BACKUP_FILE);
    match read_project(&project_path) {
        Ok(project) => Ok(project),
        Err(_) if backup_path.is_file() => {
            let project = read_project(&backup_path)?;
            // Restore the last complete snapshot so the next load does not
            // depend on the recovery sidecar.
            fs::copy(&backup_path, &project_path)?;
            Ok(project)
        }
        Err(_) if !project_path.is_file() => Err(AppError::ProjectNotFound),
        Err(primary_error) => Err(primary_error),
    }
}

fn read_project(path: &Path) -> Result<Project, AppError> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

pub fn save_project_atomic(project_directory: &Path, project: &Project) -> Result<(), AppError> {
    fs::create_dir_all(project_directory)?;
    let project_path = project_directory.join("project.json");
    let temporary_path = project_directory.join("project.json.tmp");
    let json = serde_json::to_string_pretty(project)?;
    fs::write(&temporary_path, format!("{json}\n"))?;

    if project_path.is_file() {
        // Windows rename cannot replace an existing destination. Preserve a
        // complete prior snapshot before entering the remove/rename window so
        // an interrupted process restart cannot erase the project.
        fs::copy(&project_path, project_directory.join(PROJECT_BACKUP_FILE))?;
    }

    match fs::rename(&temporary_path, &project_path) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            // Windows cannot replace an existing file with rename; retain the
            // temp-file-first behavior and only use this compatibility fallback.
            if project_path.exists() {
                fs::remove_file(&project_path)?;
                fs::rename(&temporary_path, &project_path)?;
                Ok(())
            } else {
                Err(rename_error.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::model::{SourceVideo, VideoMetadata, WatermarkConfig};

    fn sample_project() -> Project {
        Project {
            version: 1,
            id: "test-project".to_string(),
            source: SourceVideo {
                path: "C:\\videos\\test.mp4".to_string(),
                file_name: "test.mp4".to_string(),
            },
            video: VideoMetadata {
                width: 1080,
                height: 1920,
                duration_seconds: 20.021,
                fps: 30000.0 / 1001.0,
                frame_count: 601,
                codec: Some("h264".to_string()),
                pixel_format: Some("yuv420p".to_string()),
            },
            watermark: WatermarkConfig {
                template_padding: 4,
                ..Default::default()
            },
            anchors: Vec::new(),
            tracking: None,
            removal: None,
        }
    }

    #[test]
    fn project_round_trips_as_camel_case_json() {
        let project = sample_project();
        let json = serde_json::to_string(&project).expect("serialization should work");
        assert!(json.contains("durationSeconds"));
        assert!(json.contains("templatePadding"));
        let restored: Project = serde_json::from_str(&json).expect("deserialization should work");
        assert_eq!(restored.id, project.id);
        assert_eq!(restored.video.frame_count, 601);
    }

    #[test]
    fn watermark_config_defaults_template_padding_for_older_projects() {
        let restored: WatermarkConfig = serde_json::from_str(r#"{"label":"Learna AI"}"#)
            .expect("older watermark config should deserialize");
        assert_eq!(restored.template_padding, 4);
    }

    #[test]
    fn atomic_project_path_creates_project_json() {
        let directory =
            std::env::temp_dir().join(format!("watermark-studio-test-{}", uuid::Uuid::new_v4()));
        save_project_atomic(&directory, &sample_project()).expect("project should save");
        assert!(directory.join("project.json").is_file());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn load_recovers_last_good_snapshot_when_primary_is_missing() {
        let app_data =
            std::env::temp_dir().join(format!("watermark-studio-test-{}", uuid::Uuid::new_v4()));
        let directory = project_directory(&app_data, "test-project").unwrap();
        let mut original = sample_project();
        save_project_atomic(&directory, &original).expect("first project save should work");
        original.watermark.label = Some("Learna AI".to_string());
        save_project_atomic(&directory, &original).expect("second project save should work");
        fs::remove_file(directory.join("project.json")).expect("simulate interrupted replace");

        let recovered = load_project(&app_data, "test-project").expect("backup should recover");

        assert!(directory.join("project.json").is_file());
        assert_eq!(recovered.watermark.label.as_deref(), None);
        let _ = fs::remove_dir_all(app_data);
    }
}
