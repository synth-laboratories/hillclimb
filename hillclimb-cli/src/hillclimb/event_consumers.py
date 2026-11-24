"""Durable, read-only projections of Hillclimb events."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import Event as StopEvent, Thread
import time
from typing import Any

Json = dict[str, Any]


@dataclass(frozen=True)
class Consumer:
    stream: str
    inputs: tuple[str, ...]
    schema: str
    reduce: Callable[[Json, Json], tuple[Json, Json | None]]


class EventStore:
    """A run-local journal. Positions order inputs across streams and are SSE IDs."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        with closing(self.connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, stream TEXT NOT NULL, schema TEXT NOT NULL,
                    run_id TEXT NOT NULL, payload TEXT NOT NULL, cursors TEXT NOT NULL,
                    source_id TEXT, UNIQUE(run_id, stream, source_id)
                );
                CREATE INDEX IF NOT EXISTS stream_events ON events(run_id, stream, seq);
                CREATE TABLE IF NOT EXISTS consumers (
                    run_id TEXT NOT NULL, stream TEXT NOT NULL, schema TEXT NOT NULL,
                    seq INTEGER NOT NULL, state TEXT NOT NULL, cursors TEXT NOT NULL,
                    PRIMARY KEY(run_id, stream, schema)
                );
            """)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=0.25)

    def publish(self, stream: str, schema: str, payload: Json, *, source_id: str | None = None) -> None:
        with closing(self.connect()) as db, db:
            self._insert(db, stream, schema, payload, {}, source_id)

    def _insert(self, db, stream, schema, payload, cursors, source_id) -> None:
        db.execute(
            "INSERT INTO events(stream,schema,run_id,payload,cursors,source_id) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(run_id,stream,source_id) DO NOTHING",
            (stream, schema, self.run_id, json.dumps(payload, allow_nan=False),
             json.dumps(cursors), source_id),
        )

    def _events(self, db, streams: tuple[str, ...], after: int, limit: int) -> list[Json]:
        params = ",".join("?" for _ in streams)
        rows = db.execute(
            f"SELECT seq,stream,schema,payload,cursors FROM events "
            f"WHERE run_id=? AND seq>? AND stream IN ({params}) ORDER BY seq LIMIT ?",
            (self.run_id, after, *streams, limit),
        )
        return [
            {"run_id": self.run_id, "producer": stream, "schema": schema, "seq": seq,
             "payload": json.loads(payload), "input_cursors": json.loads(cursors)}
            for seq, stream, schema, payload, cursors in rows
        ]

    def events(self, stream: str, after: int = 0, *, limit: int = 128) -> list[Json]:
        with closing(self.connect()) as db:
            return self._events(db, (stream,), after, limit)

    def sse(self, stream: str, after: int = 0, *, stop: StopEvent | None = None) -> Iterator[str]:
        while True:
            events = self.events(stream, after)
            if not events:
                if stop is not None and stop.is_set():
                    return
                time.sleep(0.25)
                yield ": idle\n\n"
            for event in events:
                after = event["seq"]
                yield f"id: {after}\nevent: hillclimb.derived.v1\ndata: {json.dumps(event)}\n\n"

    def _checkpoint(self, db, consumer: Consumer) -> tuple[int, Json, Json]:
        row = db.execute(
            "SELECT seq,state,cursors FROM consumers WHERE run_id=? AND stream=? AND schema=?",
            (self.run_id, consumer.stream, consumer.schema),
        ).fetchone()
        return (row[0], json.loads(row[1]), json.loads(row[2])) if row else (0, {}, {})

    def advance(self, consumer: Consumer) -> bool:
        with closing(self.connect()) as db:
            after, state, cursors = self._checkpoint(db, consumer)
            events = self._events(db, consumer.inputs, after, 1)
        if not events:
            return False
        event = events[0]
        state, payload = consumer.reduce(state, event)
        cursors[event["producer"]] = event["seq"]
        with closing(self.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            with db:
                if self._checkpoint(db, consumer)[0] != after:
                    return True
                if payload is not None:
                    self._insert(db, consumer.stream, consumer.schema, payload, cursors,
                                 f'{consumer.schema}:{event["seq"]}')
                db.execute(
                    "INSERT INTO consumers VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(run_id,stream,schema) DO UPDATE SET "
                    "seq=excluded.seq,state=excluded.state,cursors=excluded.cursors",
                    (self.run_id, consumer.stream, consumer.schema, event["seq"],
                     json.dumps(state, allow_nan=False), json.dumps(cursors)),
                )
        return True

    def caught_up(self, consumer: Consumer) -> bool:
        with closing(self.connect()) as db:
            after, _, _ = self._checkpoint(db, consumer)
            return not self._events(db, consumer.inputs, after, 1)


def validate_graph(consumers: Sequence[Consumer]) -> None:
    graph = {consumer.stream: consumer.inputs for consumer in consumers}
    if len(graph) != len(consumers) or any(not inputs for inputs in graph.values()):
        raise ValueError("Consumers need unique output streams and at least one input")
    pending = set(graph)
    while pending:
        ready = {stream for stream in pending if pending.isdisjoint(graph[stream])}
        if not ready:
            raise ValueError("Consumer stream dependencies contain a cycle")
        pending.difference_update(ready)


class ConsumerWorkers:
    def __init__(self, store: EventStore, consumers: Sequence[Consumer], on_error: Callable) -> None:
        validate_graph(consumers)
        self.store = store
        self.consumers = consumers
        self.on_error = on_error
        self.stop = StopEvent()
        self.threads: list[Thread] = []

    def _work(self, consumer: Consumer) -> None:
        failed = False
        while not self.stop.is_set():
            try:
                advanced = self.store.advance(consumer)
                failed = False
                if not advanced:
                    self.stop.wait(0.05)
            except Exception as error:
                if not failed:
                    self.on_error(consumer.stream, error)
                    failed = True
                self.stop.wait(0.25)

    def __enter__(self):
        for consumer in self.consumers:
            thread = Thread(target=self._work, args=(consumer,), daemon=True, name=consumer.stream)
            thread.start()
            self.threads.append(thread)
        return self

    def drain(self, timeout: float = 2) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if all(self.store.caught_up(consumer) for consumer in self.consumers):
                    return True
            except sqlite3.Error:
                pass
            time.sleep(0.05)
        return False

    def __exit__(self, *_):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=0.5)


def mount_sse(app, store: EventStore, streams: Sequence[str], *, stop: StopEvent | None = None):
    """Serve derived streams on a host that already uses FastAPI."""
    from fastapi import HTTPException, Request
    from fastapi.responses import StreamingResponse

    allowed = frozenset(streams)

    @app.get("/v1/derived/{stream}")
    def subscribe(stream: str, request: Request, after_seq: int | None = None):
        if stream not in allowed:
            raise HTTPException(404, "Unknown derived stream")
        header = request.headers.get("last-event-id")
        try:
            last = int(header) if header is not None else after_seq or 0
        except ValueError as error:
            raise HTTPException(400, "Invalid cursor") from error
        if last < 0 or last >= 2**63 or (after_seq is not None and last != after_seq):
            raise HTTPException(400, "Invalid cursor")
        return StreamingResponse(store.sse(stream, last, stop=stop), media_type="text/event-stream")
