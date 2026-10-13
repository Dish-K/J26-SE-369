# Database migrations

Schema for CodeTrace. Apply the numbered files in order on an empty database.
008 is part of the sequence, so the schema is not complete until it has run.

## Files

| File | What it does |
|---|---|
| `001_extensions.sql` | TimescaleDB and UUID extensions |
| `002_tables.sql` | C1 capture tables, task submissions and feature vectors |
| `003_hypertables.sql` | Makes `events` a hypertable (the only one) |
| `004_indexes.sql` | Indexes for event extraction and clock sync |
| `005_classifier_output.sql` | C2 window predictions, flags and task scores |
| `006_classifier_flag_final_locations.sql` | Maps each flag to the submitted code, plus the view C3 reads |
| `007_code_snapshots.sql` | Periodic full-document snapshots |
| `008_snapshot_version_and_retention.sql` | Adds `load_seq` to snapshot and submission identity, adds `capture_reason` |

The three `verify_*.sql` scripts test the schema inside a transaction and roll
back, so they leave no rows behind:

- `verify_contract.sql` - capture and C2 output
- `verify_snapshots.sql` - snapshot identity, dedup, reloads and redaction
- `verify_retention.sql` - both stages of the retention sweep

`verify_snapshots.sql` and `verify_retention.sql` fail if 008 has not been
applied. That is expected: they test the bugs 008 fixes.

## Applying

```bash
cd backend/app/db/migrations
for f in 0*.sql; do
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f" || break
done
for f in verify_contract.sql verify_snapshots.sql verify_retention.sql; do
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f" || break
done
```

On a fresh TimescaleDB container, wait until `select 1` succeeds a few times in
a row before applying. The container restarts once while it initialises, and a
migration run during that restart fails with "the database system is shutting
down".

After a clean run there are 13 tables and 1 view, and `events` is the only hypertable.

## C2 rules the schema relies on

- Live windows are 30 seconds wide, emitted every 10 seconds.
- `feature_vectors.feature_vector_id` identifies one model input. A model
  version scores each feature vector at most once.
- A prediction stores all three class probabilities. `confidence` is the
  probability of the predicted class.
- `ai_positive` is true only when an AI class wins and its confidence meets the
  threshold stored in that model version's artifact.
- Paste, suggestion, timing and edit observations are evidence. They are never
  rules that decide a class on their own.
- AI-positive windows join into one flag only when the task, model version and
  AI category match, the windows are consecutive, and the code ranges overlap or
  touch. Separate ranges in one window stay separate. A no_ai or below-threshold
  window ends the run.
- A merged flag copies the numbers of its highest-confidence window. Probabilities
  are never averaged.
- Flag line ranges are historical, from when the flag was raised.
  `classifier_flag_final_locations` records whether each one is `mapped`,
  `deleted` or `unmappable` in the submitted code.
- C3 reads `classifier_flags_for_probing`, which shows only mapped flags with
  their final line ranges.
- A task's `ai_confidence` is the highest confidence among its flags. A task
  with no flags is `no_ai_detected` with a NULL confidence.
- Two separate episodes can flag exactly the same lines, so the duplicate check
  on flags includes the detection time.

The keys, types and ranges inside `feature_vectors.features` are not defined in
SQL. They belong to the shared feature schema in `shared_features/`.

## Changing the schema

Once a numbered file has been applied to a shared database, leave it as it is
and put the change in a new file. 007 had a wrong unique key, and 008 fixes it
this way.

The next number, `009_classifier_flag_source_context.sql`, is reserved for C2.

When you add a table, add rows for it to the fixture in `verify_retention.sql`
in the same change, and check the script fails until the sweep handles it.

## Before tagging pilot-v1

- All numbered migrations run on an empty database with `ON_ERROR_STOP=1`.
- All three verify scripts pass.
- A real sample session and its events can be stored and read back in order.
- A feature vector is validated against the shared schema before it is inserted.
- C2 can store a window prediction, a multi-window flag and a clean task score.
- C3 reads only final mapped ranges, never the historical ones.
