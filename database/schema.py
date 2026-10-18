
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_hash TEXT NOT NULL UNIQUE,
    config_hash TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    dataset TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    attack_hash TEXT NOT NULL,
    attack_name TEXT,
    defense_hash TEXT NOT NULL,
    defense_name TEXT,
    outcome TEXT,
    best_reconstruction_hash TEXT,
    raw_description TEXT NOT NULL,
    run_metrics TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    parameters TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
    UNIQUE(run_id, step)
);

CREATE TABLE IF NOT EXISTS images (
    image_hash TEXT PRIMARY KEY,
    run_id INTEGER NOT NULL,
    image_type TEXT NOT NULL,
    image_data BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_run_hash ON runs(run_hash);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset);
CREATE INDEX IF NOT EXISTS idx_runs_attack_hash ON runs(attack_hash);
CREATE INDEX IF NOT EXISTS idx_runs_attack_name ON runs(attack_name);
CREATE INDEX IF NOT EXISTS idx_runs_defense_hash ON runs(defense_hash);
CREATE INDEX IF NOT EXISTS idx_runs_defense_name ON runs(defense_name);
CREATE INDEX IF NOT EXISTS idx_components_run ON components(run_id);
CREATE INDEX IF NOT EXISTS idx_components_category_name ON components(category, name);
CREATE INDEX IF NOT EXISTS idx_logs_run_step ON logs(run_id, step);
CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id);
"""


def migrate_schema(connection):
    """Add result metadata without rewriting or deleting historical rows."""
    connection.executescript(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    for name, definition in (
        ("config_hash", "TEXT"),
        ("attempt", "INTEGER NOT NULL DEFAULT 1"),
        ("summary", "TEXT NOT NULL DEFAULT '{}'"),
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash)")
    connection.commit()
