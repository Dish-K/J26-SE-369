-- Only events is a hypertable. The smaller tables stay ordinary so their unique
-- keys do not have to include the time column.
SELECT create_hypertable(
  'events',
  'time',
  chunk_time_interval => INTERVAL '1 hour'
);
