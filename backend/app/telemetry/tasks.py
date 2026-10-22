from uuid import UUID

import asyncpg

from backend.app.telemetry.ingest import Submit

_CLOSE_TASK = """
UPDATE tasks SET status = 'submitted', submitted_at = now()
WHERE task_id = $1 AND session_id = $2 AND status = 'active'
RETURNING submitted_at
"""

_STORE_SUBMISSION = """
INSERT INTO task_submissions (task_id, session_id, code_text, version_id, submitted_at, load_seq)
VALUES ($1, $2, $3, $4, $5, $6)
"""

_ALREADY_SUBMITTED = "SELECT 1 FROM task_submissions WHERE task_id = $1 AND session_id = $2"

async def submit_task(
    connection: asyncpg.Connection, session_id: UUID, load_seq: int, submission: Submit
) -> bool:
    async with connection.transaction():
        submitted_at = await connection.fetchval(_CLOSE_TASK, submission.task_id, session_id)
        if submitted_at is None:
            # A resend after a lost reply gets the same answer; any other closed task is refused.
            return await connection.fetchval(
                _ALREADY_SUBMITTED, submission.task_id, session_id
            ) is not None

        await connection.execute(
            _STORE_SUBMISSION,
            submission.task_id,
            session_id,
            submission.code_text,
            submission.version_id,
            submitted_at,
            load_seq,
        )
        return True
