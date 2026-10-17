from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Same kinds as events_task_scoped_have_task_chk, checked here so the error names the field.
TASK_SCOPED_KINDS = frozenset({0, 1, 2, 8, 9})
TYPING_KINDS = frozenset({0, 1, 2})

class Event(BaseModel):
    # time and gap_ms are set by the server, so the browser may not send them.
    model_config = ConfigDict(extra="forbid")

    seq_in_batch: int = Field(ge=0)
    kind: int = Field(ge=0, le=14)
    perf_now: float = Field(ge=0)
    task_id: UUID | None = None

    key_class: int | None = None
    key_code: str | None = None
    modifiers: int | None = None
    is_repeat: bool | None = None
    is_trusted: bool | None = None

    origin: int | None = None
    version_id: int | None = None
    change_index: int | None = None
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    range_offset: int | None = None
    inserted_len: int | None = None
    removed_len: int | None = None
    inserted_lines: int | None = None
    doc_len: int | None = None
    paste_origin: int | None = None
    inserted_text: str | None = None

    env_state: int | None = None
    viewport_w: int | None = None
    viewport_h: int | None = None

    run_id: UUID | None = None
    exit_status: int | None = None
    browser_duration_ms: float | None = None
    sandbox_time_ms: float | None = None
    sandbox_memory_kb: int | None = None

    copied_len: int | None = None
    source_pane: int | None = None

    @model_validator(mode="after")
    def check_task_id(self) -> "Event":
        if self.kind in TASK_SCOPED_KINDS and self.task_id is None:
            raise ValueError(f"kind {self.kind} events need a task_id")
        return self

class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    load_seq: int = Field(ge=0)
    batch_seq: int = Field(ge=0)
    events: list[Event] = Field(min_length=1)

# Read from Event, so the row values and the COPY column list cannot drift apart.
CLIENT_COLUMNS = tuple(Event.model_fields)
EVENT_COLUMNS = ("time", "session_id", "load_seq", "batch_seq", "gap_ms", *CLIENT_COLUMNS)

def event_rows(batch: Batch, previous_typing: dict[UUID, float], offset_ms: float) -> list[tuple]:
    last_typing = dict(previous_typing)
    rows = []

    # The array is capture order, which is the order gaps must be measured in.
    for event in batch.events:
        gap_ms = None
        # Only typing events carry a gap, so a tab switch does not split a pause.
        if event.kind in TYPING_KINDS:
            before = last_typing.get(event.task_id)
            if before is not None:
                gap_ms = event.perf_now - before
            last_typing[event.task_id] = event.perf_now

        # With a zero offset (until the Step 5 clock handshake) this reads as 1970.
        time = datetime.fromtimestamp((event.perf_now + offset_ms) / 1000, tz=UTC)
        values = [getattr(event, column) for column in CLIENT_COLUMNS]
        rows.append((time, batch.session_id, batch.load_seq, batch.batch_seq, gap_ms, *values))

    return rows
