"""Detachable Temper transcript and declarative optimizer Canvas."""

from __future__ import annotations

import asyncio
import httpx
import json
import shutil
import sqlite3
from contextlib import closing
from collections import deque
from pathlib import Path

from mlok_temper.ui.canvas import Canvas
from mlok_temper.ui.panel_split import PanelBoundary, PanelSplit
from prompt_toolkit import Application
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from .client import Client, HillclimbError, Run
from .operator import Operator
from .visuals import (
    _replace,
    lens_name,
    lens_registry,
    validate_view,
    view_lock,
    write_visuals,
)

__all__ = ["attach", "board"]


class Board:
    def __init__(self, client, status):
        self.client = client
        self.algorithm = status["algorithm_id"]
        self.directory = Path(status["directory"]) / "observer" / "visuals"
        self.canvas = Canvas()
        self.revision = None
        self.inbox = []
        self.view = self.directory / "view.active.json"

    def refresh(self):
        bundle = self.client.visuals()
        self.inbox = bundle["edits"].get("inbox", [])
        queue_seq = max(
            (r["resolution_seq"] or r["admitted_seq"] for r in self.inbox), default=0
        )
        revision = (
            bundle["core"]["event_seq"],
            bundle["core"]["state_rail"]["current"],
            queue_seq,
            tuple(
                (r["role"], r["count"]) for r in bundle["core"].get("active_agents", [])
            ),
            bundle["core"]["summary_metrics"].get("active_rollouts"),
        )
        if revision != self.revision:
            write_visuals(self.directory, bundle, algorithm=self.algorithm)
            self.revision = revision
        self.canvas.load(str(self.view))
        if self.canvas.error:
            raise ValueError(self.canvas.error)

    def lines(self, width):
        return [
            line for line, _ in self.canvas.lines({}, self.revision, max(20, width))
        ]

    def edit(self, line):
        if line == "/reset-board":
            with view_lock(self.directory):
                _replace(
                    self.directory,
                    self.view.name,
                    (self.directory / "view.base.json").read_bytes(),
                )
            return "Restored base board"
        if line.startswith("/save-lens "):
            target = json.loads((self.directory / "target.json").read_bytes())[
                "target_id"
            ]
            lens = lens_name(self.algorithm, target)
            if line.split(maxsplit=1)[1] != lens:
                raise ValueError("Name must match the active optimizer and target")
            with view_lock(self.directory):
                payload = self.view.read_bytes()
                validate_view(payload)
            registry = lens_registry()
            registry.mkdir(parents=True, exist_ok=True, mode=0o700)
            _replace(registry, lens + ".canvas.json", payload)
            return f"Saved {lens}"
        return None


def save_board_png(lines, output):
    from math import ceil
    from PIL import Image, ImageDraw, ImageFont
    from rich.cells import cell_len

    for family in ("DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            font = ImageFont.truetype(family, 16)
            break
        except OSError:
            continue
    else:
        font = ImageFont.load_default(size=16)
    cell = ceil(font.getlength("M"))
    width = max((cell_len(line.plain) for line in lines), default=1)
    picture = Image.new("RGB", (width * cell + 24, len(lines) * 22 + 20), "#15181d")
    draw = ImageDraw.Draw(picture)
    console = Console(color_system="truecolor")
    for row, line in enumerate(lines):
        x, y = 12, 10 + 22 * row
        for segment in line.render(console):
            style = segment.style
            color = style.color.get_truecolor() if style and style.color else "#e6e8ed"
            for char in segment.text:
                dots = ord(char) - 0x2800
                if 0 <= dots < 256:
                    for bit, (col, level) in enumerate(
                        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3))
                    ):
                        if dots & (1 << bit):
                            left, top = x + 1 + col * (cell // 2), y + 3 + level * 4
                            draw.ellipse((left, top, left + 2, top + 2), fill=color)
                elif char.isprintable() and char != " ":
                    draw.text((x, y), char, font=font, fill=color)
                x += cell_len(char) * cell
    picture.save(output, format="PNG")


def board(run: Run, watch: bool, png: Path | None = None):
    import time

    console = Console()
    with Client(run) as client:
        display = Board(client, client.status())
        while True:
            display.refresh()
            if png is not None:
                save_board_png(display.lines(112), png)
                return
            if watch and console.is_terminal:
                console.clear()
            for line in display.lines(console.width):
                console.print(line)
            if not watch or display.revision[1] in {"completed", "cancelled", "failed"}:
                return
            time.sleep(1)


class Divider:
    def __init__(self):
        self.panel = {"visible": True}
        self.panel_expanded = self.panel_dragging = False
        self.panel_boundary_x = 0
        self.columns = None

    def resize_panel(self, columns):
        self.columns = max(24, min(columns, shutil.get_terminal_size().columns - 22))


async def _attach(run, after_seq=0):
    from .cli import _compose

    with Client(run) as client:
        status = client.status()
        display = Board(client, status)
        operator = Operator(
            run, Path(status["directory"]), client, status.get("optimizer_commands", {})
        )
        divider = Divider()
        transcript = deque(maxlen=100)
        observer = display.directory.parent
        observer.mkdir(parents=True, exist_ok=True, mode=0o700)
        history = observer / "transcript.sqlite"
        with closing(sqlite3.connect(history)) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS messages(seq INTEGER PRIMARY KEY, speaker TEXT, text TEXT)"
            )
            transcript.extend(
                f"{who}  {text}"
                for who, text in reversed(
                    db.execute(
                        "SELECT speaker,text FROM messages ORDER BY seq DESC LIMIT 100"
                    ).fetchall()
                )
            )
        entry = TextArea(height=1, multiline=False, prompt="> ")
        board_pane = TextArea(
            read_only=True,
            scrollbar=True,
            focusable=True,
            wrap_lines=False,
            width=lambda: Dimension(preferred=divider.columns, weight=2),
        )
        log = TextArea(text="\n\n".join(transcript), read_only=True, scrollbar=True)
        keys = KeyBindings()
        active = None

        def append(who, text):
            if who != "Queue":
                with closing(sqlite3.connect(history)) as db, db:
                    db.execute(
                        "INSERT INTO messages(speaker,text) VALUES(?,?)",
                        (who, text[:16000]),
                    )
                    db.execute(
                        "DELETE FROM messages WHERE seq <= (SELECT max(seq)-100 FROM messages)"
                    )
            transcript.append(f"{who}  {text}")
            log.text = "\n\n".join(transcript)
            log.buffer.cursor_position = len(log.text)

        async def respond(line):
            try:
                if line.startswith("/"):
                    text = display.edit(line)
                    if text is None:
                        text = await asyncio.to_thread(_compose, client, line)
                else:
                    text = await operator.chat(line)
                append("Operator", text)
            except (ValueError, HillclimbError, OSError, TimeoutError) as error:
                await operator.close()
                append("Operator", str(error))

        @keys.add("enter", filter=has_focus(entry))
        def submit(event):
            nonlocal active
            line = entry.text.strip()
            if line == "/detach":
                event.app.exit()
            elif line and (active is None or active.done()):
                entry.text = ""
                append("You", line)
                active = event.app.create_background_task(respond(line))

        @keys.add("c-d")
        def detach(event):
            event.app.exit()

        @keys.add("tab")
        def focus(event):
            event.app.layout.focus(
                entry if event.app.layout.has_focus(board_pane) else board_pane
            )

        heading = Window(
            FormattedTextControl(
                lambda: (
                    f"{status['algorithm_id']}   {status['run_id']}   {display.revision[1] if display.revision else status['status']}"
                )
            ),
            height=1,
        )
        split = PanelSplit(
            divider,
            [
                HSplit(
                    [Window(FormattedTextControl("Operator"), height=1), log, entry]
                ),
                PanelBoundary(divider),
                board_pane,
            ],
        )
        app = Application(
            layout=Layout(HSplit([heading, split]), focused_element=entry),
            key_bindings=keys,
            full_screen=True,
            mouse_support=True,
        )

        dirty = asyncio.Event()
        dirty.set()

        async def events():
            cursor = after_seq or status["event_seq"]
            transport = httpx.AsyncHTTPTransport(uds=str(run.socket))
            async with httpx.AsyncClient(
                transport=transport, base_url="http://localhost", trust_env=False
            ) as reader:
                while True:
                    try:
                        async with reader.stream(
                            "GET", "/v1/events", headers={"Last-Event-ID": str(cursor)}
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if line.startswith("id:"):
                                    cursor = int(line[3:])
                                    dirty.set()
                    except httpx.HTTPError as error:
                        board_pane.text = str(error)
                    await asyncio.sleep(1)
                    dirty.set()

        async def refresh():
            observed = {}

            while True:
                try:
                    if dirty.is_set():
                        dirty.clear()
                        await asyncio.to_thread(display.refresh)
                    else:
                        display.canvas.load(str(display.view))
                    for receipt in reversed(display.inbox):
                        ident, state = receipt["op_id"], receipt["status"]
                        if observed.get(ident) != state:
                            append(
                                "Queue",
                                f"{ident}  {state}"
                                + (
                                    f"  {receipt['reason']}"
                                    if receipt.get("reason")
                                    else ""
                                ),
                            )
                    observed = {r["op_id"]: r["status"] for r in display.inbox}
                    width = divider.columns or (app.output.get_size().columns // 2)
                    board_pane.text = "\n".join(
                        line.plain for line in display.lines(width - 2)
                    )
                except (ValueError, HillclimbError, OSError) as error:
                    board_pane.text = str(error)
                await asyncio.sleep(1)

        def start():
            app.create_background_task(events())
            app.create_background_task(refresh())

        try:
            await app.run_async(pre_run=start)
        finally:
            await operator.close()


def attach(run, after_seq=0):
    asyncio.run(_attach(run, after_seq))
