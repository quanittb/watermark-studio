use crate::error::AppError;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use uuid::Uuid;

const LEGACY_JOBS_FILE: &str = "jobs.json";
const DATABASE_FILE: &str = "watermark-studio.db";
static RECOVERY_DONE: OnceLock<()> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStatus {
    Imported,
    Scanning,
    AwaitingReview,
    Ready,
    Queued,
    Preparing,
    Inferencing,
    Encoding,
    Verifying,
    Completed,
    NeedsReview,
    Failed,
    Canceled,
    Interrupted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRecord {
    pub id: String,
    pub project_id: String,
    pub source_name: String,
    pub output_root: Option<String>,
    #[serde(default)]
    pub output_name: Option<String>,
    pub output_path: Option<String>,
    pub status: JobStatus,
    pub stage: String,
    pub progress: f64,
    #[serde(default)]
    pub batch_progress: f64,
    #[serde(default)]
    pub current_frame: Option<u64>,
    #[serde(default)]
    pub current_chunk: Option<u64>,
    #[serde(default)]
    pub elapsed_seconds: Option<u64>,
    #[serde(default)]
    pub eta_seconds: Option<u64>,
    #[serde(default)]
    pub replacement_config: Option<serde_json::Value>,
    #[serde(default)]
    pub hardware_profile: Option<String>,
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    #[serde(default)]
    pub qa_report_path: Option<String>,
    #[serde(default)]
    pub contact_sheet_path: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

fn default_attempt() -> u32 {
    1
}

pub fn database_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("WatermarkStudio").join(DATABASE_FILE)
}

fn legacy_jobs_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("WatermarkStudio").join(LEGACY_JOBS_FILE)
}

fn open_database(app_data_dir: &Path) -> Result<Connection, AppError> {
    let path = database_path(app_data_dir);
    let directory = path
        .parent()
        .ok_or_else(|| AppError::Io("Invalid database directory.".to_string()))?;
    fs::create_dir_all(directory)?;
    let connection = Connection::open(path).map_err(|error| AppError::Io(error.to_string()))?;
    connection
        .busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|error| AppError::Io(error.to_string()))?;
    connection.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA foreign_keys=ON;
         CREATE TABLE IF NOT EXISTS projects (
           id TEXT PRIMARY KEY, source_path TEXT NOT NULL, source_name TEXT NOT NULL,
           output_root TEXT, output_name TEXT, calibration_status TEXT NOT NULL DEFAULT 'IMPORTED',
           created_at TEXT NOT NULL, updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS jobs (
           id TEXT PRIMARY KEY, project_id TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL,
           progress REAL NOT NULL, output_path TEXT, hardware_profile TEXT, attempt INTEGER NOT NULL,
           qa_report_path TEXT, contact_sheet_path TEXT, error_code TEXT, record_json TEXT NOT NULL,
           created_at TEXT NOT NULL, updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS job_attempts (
           id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, attempt INTEGER NOT NULL,
           status TEXT NOT NULL, started_at TEXT, finished_at TEXT, error_code TEXT, UNIQUE(job_id, attempt)
         );
         CREATE TABLE IF NOT EXISTS job_events (
           id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, attempt INTEGER NOT NULL,
           status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL, payload_json TEXT,
           created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS artifacts (
           id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, attempt INTEGER NOT NULL,
           kind TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT, created_at TEXT NOT NULL,
           UNIQUE(job_id, attempt, kind, path)
         );"
    ).map_err(|error| AppError::Io(error.to_string()))?;
    migrate_legacy_json(&connection, app_data_dir)?;
    if RECOVERY_DONE.set(()).is_ok() {
        connection.execute(
            "UPDATE jobs SET status='INTERRUPTED', stage='Interrupted by application restart', updated_at=?1,
             record_json=json_set(record_json, '$.status', 'INTERRUPTED', '$.stage', 'Interrupted by application restart', '$.updatedAt', ?1)
             WHERE status IN ('PREPARING','INFERENCING','ENCODING','VERIFYING')",
            params![chrono_like_now()],
        ).map_err(|error| AppError::Io(error.to_string()))?;
    }
    Ok(connection)
}

fn migrate_legacy_json(connection: &Connection, app_data_dir: &Path) -> Result<(), AppError> {
    let count: i64 = connection
        .query_row("SELECT COUNT(*) FROM jobs", [], |row| row.get(0))
        .map_err(|error| AppError::Io(error.to_string()))?;
    let legacy = legacy_jobs_path(app_data_dir);
    if count != 0 || !legacy.is_file() {
        return Ok(());
    }
    let records: Vec<JobRecord> = serde_json::from_str(&fs::read_to_string(&legacy)?)?;
    for record in &records {
        upsert_record(connection, record)?;
    }
    let backup = legacy.with_file_name("jobs.json.migrated.bak");
    if !backup.exists() {
        fs::copy(legacy, backup)?;
    }
    Ok(())
}

fn status_text(status: JobStatus) -> Result<String, AppError> {
    serde_json::to_value(status)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| AppError::Json("Invalid job status.".to_string()))
}

fn upsert_record(connection: &Connection, record: &JobRecord) -> Result<(), AppError> {
    let body = serde_json::to_string(record)?;
    connection.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,output_path,hardware_profile,attempt,qa_report_path,contact_sheet_path,error_code,record_json,created_at,updated_at)
         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)
         ON CONFLICT(id) DO UPDATE SET project_id=excluded.project_id,status=excluded.status,stage=excluded.stage,
         progress=excluded.progress,output_path=excluded.output_path,hardware_profile=excluded.hardware_profile,
         attempt=excluded.attempt,qa_report_path=excluded.qa_report_path,contact_sheet_path=excluded.contact_sheet_path,
         error_code=excluded.error_code,record_json=excluded.record_json,updated_at=excluded.updated_at",
        params![record.id, record.project_id, status_text(record.status)?, record.stage, record.progress,
            record.output_path, record.hardware_profile, record.attempt, record.qa_report_path,
            record.contact_sheet_path, record.error_code, body, record.created_at, record.updated_at],
    ).map_err(|error| AppError::Io(error.to_string()))?;
    Ok(())
}

pub fn load(app_data_dir: &Path) -> Result<Vec<JobRecord>, AppError> {
    let connection = open_database(app_data_dir)?;
    let mut statement = connection
        .prepare("SELECT record_json FROM jobs ORDER BY CAST(created_at AS INTEGER), rowid")
        .map_err(|error| AppError::Io(error.to_string()))?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| AppError::Io(error.to_string()))?;
    let mut records = Vec::new();
    for body in rows {
        records.push(serde_json::from_str(
            &body.map_err(|error| AppError::Io(error.to_string()))?,
        )?);
    }
    Ok(records)
}

pub fn save(app_data_dir: &Path, records: &[JobRecord]) -> Result<(), AppError> {
    let mut connection = open_database(app_data_dir)?;
    let transaction = connection
        .transaction()
        .map_err(|error| AppError::Io(error.to_string()))?;
    transaction
        .execute("DELETE FROM jobs", [])
        .map_err(|error| AppError::Io(error.to_string()))?;
    for record in records {
        upsert_record(&transaction, record)?;
    }
    transaction
        .commit()
        .map_err(|error| AppError::Io(error.to_string()))
}

pub fn new_record(
    project_id: String,
    source_name: String,
    output_root: Option<String>,
    status: JobStatus,
) -> JobRecord {
    let now = chrono_like_now();
    JobRecord {
        id: Uuid::new_v4().to_string(),
        project_id,
        source_name,
        output_root,
        output_name: None,
        output_path: None,
        status,
        stage: "Queued".to_string(),
        progress: 0.0,
        batch_progress: 0.0,
        current_frame: None,
        current_chunk: None,
        elapsed_seconds: None,
        eta_seconds: None,
        replacement_config: None,
        hardware_profile: None,
        attempt: 1,
        qa_report_path: None,
        contact_sheet_path: None,
        error_code: None,
        error: None,
        created_at: now.clone(),
        updated_at: now,
    }
}

pub fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}
