-- Extraction walks events in capture order; time alone can tie or jump back after a reload.
CREATE INDEX events_extract_order_idx
  ON events (session_id, task_id, load_seq, batch_seq, seq_in_batch);

CREATE INDEX clock_session_load_idx
  ON clock_sync_samples (session_id, load_seq, time);

-- Other lookups are already covered by unique constraints or the hypertable's time index.
