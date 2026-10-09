-- 003_hypertables.sql
-- Only the high-volume raw telemetry stream is a TimescaleDB hypertable.
--
-- events:               ~4,000-6,000 rows/session and queried by time.
-- clock_sync_samples:   small ordinary research table.
-- feature_vectors:      comparatively small, versioned Contract-A outputs.
-- classifier outputs:   ordinary relational/audit data (created later).
--
-- Keeping the smaller tables ordinary avoids unnecessary TimescaleDB unique-
-- index restrictions and keeps their relational keys straightforward.

SELECT create_hypertable(
  'events',
  'time',
  chunk_time_interval => INTERVAL '1 hour'
);
