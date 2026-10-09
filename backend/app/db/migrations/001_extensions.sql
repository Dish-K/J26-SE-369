-- 001_extensions.sql
-- Shared database capabilities required by the project.
-- Run before any table/hypertable migration.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
