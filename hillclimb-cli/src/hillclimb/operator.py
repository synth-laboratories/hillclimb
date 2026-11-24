"""Temper harness tools for an attached operator. All writes use the campaign inbox or Canvas."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import tomllib
from urllib.parse import urlsplit
from contextlib import closing
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import zstandard

from .client import Client, Command, HillclimbError, Run
from .visuals import _replace, validate_view, view_lock

__all__ = ["Operator", "Inspector"]

MAX_TRACE_BYTES = 512 * 1024


class Inspector:
    def __init__(self, directory: Path):
        self.directory = directory

    def select(self, statement, params=()):
        uri = (self.directory / "ledger.sqlite").as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(statement, params)]

    def traces(self, policy_id, split):
        if split not in {"train_reflect", "val_pareto"}:
            raise ValueError("operator evidence requires a search split")
        rows = self.select(
            "SELECT trial_id,task_id,trace_cas_sha256 FROM trials WHERE policy_id=? AND split=? "
            "ORDER BY rowid DESC LIMIT 4",
            (policy_id, split),
        )
        for row in rows:
            digest = row.pop("trace_cas_sha256")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid trace digest in ledger")
            path = self.directory / "cas" / "sha256" / digest[:2] / (digest + ".zst")
            with (
                path.open("rb") as raw,
                zstandard.ZstdDecompressor().stream_reader(raw) as stream,
            ):
                data = stream.read(MAX_TRACE_BYTES + 1)
            if len(data) > MAX_TRACE_BYTES:
                row["detail"] = "Trace exceeds operator read limit"
            elif hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("trace digest mismatch")
            else:
                row["trace"] = json.loads(data)
        return rows


def _schema(**fields):
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


class Operator:
    def __init__(self, run: Run, directory: Path, client: Client, commands):
        self.run, self.directory, self.client = run, directory, client
        self.inspector = Inspector(directory)
        self.views = directory / "observer" / "visuals"
        self.backend = None
        self.thread = None
        self.commands = commands
        string = {"type": "string"}
        self.schemas = {
            "inspect_run_status": _schema(),
            "list_candidates": _schema(),
            "inspect_target_telemetry": _schema(),
            "inspect_gelo_themes_and_meta_seeds": _schema(),
            "list_command_inbox": _schema(),
            "inspect_traces": _schema(
                policy_id=string, split={"enum": ["train_reflect", "val_pareto"]}
            ),
            "diff_policies": _schema(left=string, right=string),
            "use_canvas": _schema(spec={"type": "object"}),
            "steer": _schema(note=string),
            "inject_candidate": _schema(
                parent_id=string, intervention={"type": "object"}
            ),
            "steer_proposer": _schema(
                note=string, next_proposals={"type": ["integer", "null"], "minimum": 1}
            ),
            "pin_or_ban_lever": _schema(
                lever={
                    "enum": [
                        "prompt_patch",
                        "harness_patch",
                        "env_projection_patch",
                        "model_route_patch",
                        "sft",
                        "fbc",
                        "opsd",
                        "rlvr",
                    ]
                },
                action={"enum": ["pin", "ban", "release"]},
            ),
            "cap": _schema(micros={"type": "integer", "minimum": 0}),
            **{name: _schema() for name in ("pause", "resume", "cancel")},
            **commands,
        }

    def tool(self, name, args, op_id):
        if name not in self.schemas or set(args) != set(self.schemas[name]["required"]):
            raise ValueError("unknown tool or invalid arguments")
        if name == "inspect_traces":
            return self.inspector.traces(**args)
        if name == "diff_policies":
            return self.inspector.select(
                "SELECT a.task_id,a.split,a.seed,sa.primary_score AS lhs,sb.primary_score AS rhs "
                "FROM trials a JOIN assessments sa ON sa.effect_id=a.effect_id "
                "JOIN trials b ON b.split=a.split AND b.task_id=a.task_id AND b.seed=a.seed "
                "AND b.origin_kind=a.origin_kind AND b.parent_rollout_id IS a.parent_rollout_id "
                "AND b.policy_call_index IS a.policy_call_index JOIN assessments sb ON sb.effect_id=b.effect_id "
                "WHERE a.split IN ('train_reflect','val_pareto') AND a.policy_id=? AND b.policy_id=? "
                "AND sa.primary_score!=sb.primary_score ORDER BY a.rowid,b.rowid LIMIT 40",
                (args["left"], args["right"]),
            )
        if name == "list_command_inbox":
            return self.inspector.select(
                "SELECT a.op_id,a.command_json,a.admitted_seq,r.refusal,r.resolution_seq FROM command_admissions a LEFT JOIN command_inbox r USING(op_id) WHERE a.run_id=(SELECT run_id FROM runs) ORDER BY a.admitted_seq DESC LIMIT 30"
            )
        if name == "inspect_run_status":
            return self.client.status()
        if name == "list_candidates":
            return self.client.visuals()["candidates"]
        if name == "inspect_gelo_themes_and_meta_seeds":
            return self.client.visuals()["algo"].get("themes", {})
        if name == "inspect_target_telemetry":
            return self.client.visuals().get("target", {})
        if name == "use_canvas":
            payload = json.dumps(args["spec"], allow_nan=False).encode()
            validate_view(payload)
            with view_lock(self.views):
                _replace(self.views, "view.active.json", payload)
            return {"updated": True}
        if name in {"steer_proposer", "inject_candidate", "pin_or_ban_lever"}:
            return self.client.command(
                Command(name), value=args, op_id=op_id, enqueue=True
            )
        if name in self.commands:
            return self.client.command(
                Command.ALGO, value={name: args}, op_id=op_id, enqueue=True
            )
        value = args.get("note") if name == "steer" else args.get("micros")
        return self.client.command(
            Command(name), value=value, op_id=op_id, enqueue=True
        )

    async def connect(self):
        from mlok_temper.client import Backend

        executable = shutil.which(
            os.environ.get("HILLCLIMB_TEMPER")
            or os.environ.get("MLOK_PREBUILT_RUNTIME")
            or "mlok-temper-runtime"
        )
        config = os.environ.get("HILLCLIMB_OPERATOR_CONFIG")
        if not executable or not config:
            raise HillclimbError(
                "Temper chat requires mlok-temper-runtime and HILLCLIMB_OPERATOR_CONFIG"
            )
        with Path(config).open("rb") as stream:
            profile = tomllib.load(stream)["agent"]
        endpoint = urlsplit(profile["endpoint"])
        if (
            not endpoint.path.endswith("/responses")
            or endpoint.query
            or endpoint.fragment
            or endpoint.username
            or endpoint.scheme != "https"
            and not (
                endpoint.scheme == "http"
                and endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        ):
            raise HillclimbError(
                "Operator profile requires a Responses endpoint over HTTPS or loopback HTTP"
            )
        token = os.environ.get(profile.get("api_key_env", ""))
        if not token:
            raise HillclimbError(
                "Operator profile credential is unavailable in its configured environment variable"
            )
        route = {
            "wire_api": "responses",
            "base_url": profile["endpoint"].removesuffix("/responses"),
            "http_headers": {"Authorization": "Bearer " + token},
            "request_max_retries": 0,
            "stream_max_retries": 0,
        }
        workspace = self.directory / "observer" / "temper"
        workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [
            executable,
            "app-server",
            "--harness-mode",
            "--config",
            str(Path(config).resolve()),
            "--workspace",
            str(workspace),
            "--store",
            str(workspace / "store"),
        ]
        self.backend = await Backend.process(
            command, env={**os.environ, "MLOK_HOME": str(workspace)}
        )
        await self.backend.initialize()
        thread_file = workspace / "operator.thread"
        previous = (
            {"threadId": thread_file.read_text().strip()}
            if thread_file.exists()
            else {}
        )
        started = await self.backend.request(
            "thread/resume" if previous else "thread/start",
            **previous,
            cwd=str(workspace),
            approvalPolicy="never",
            modelProvider="hillclimb",
            model=profile["model"],
            config={"model_providers.hillclimb": route},
            sandbox="read-only",
            baseInstructions="Use the supplied campaign evidence and tool receipts. Queue only operator-requested changes.",
            dynamicTools=[
                {"name": n, "description": n.replace("_", " "), "inputSchema": s}
                for n, s in self.schemas.items()
            ],
        )
        self.thread = started["thread"]["id"]
        _replace(workspace, thread_file.name, self.thread.encode())

    async def chat(self, message):
        async with asyncio.timeout(120):
            if self.thread is None:
                await self.connect()
            response = await self.backend.request(
                "turn/start",
                threadId=self.thread,
                input=[{"type": "text", "text": message}],
            )
            turn = response["turn"]["id"]
            parts = []
            completed = []
            while (frame := await self.backend.events.get()) is not None:
                method = frame.get("method")
                params = frame.get("params", {})
                same = (
                    params.get("threadId") == self.thread
                    and params.get("turnId") == turn
                )
                if "id" in frame:
                    if (
                        method != "item/tool/call"
                        or not same
                        or params.get("namespace")
                        or not params.get("callId")
                    ):
                        raise HillclimbError("unexpected Temper tool request")
                    op_id = uuid5(
                        NAMESPACE_URL,
                        f"{self.run.run_id}/{self.thread}/{turn}/{params['callId']}",
                    )
                    try:
                        result = await asyncio.to_thread(
                            self.tool, params["tool"], params["arguments"], op_id
                        )
                        content, success = (
                            json.dumps(result, ensure_ascii=False)[:64000],
                            True,
                        )
                    except (
                        ValueError,
                        OSError,
                        HillclimbError,
                        sqlite3.Error,
                        zstandard.ZstdError,
                    ) as error:
                        content, success = str(error), False
                    await self.backend.write(
                        {
                            "id": frame["id"],
                            "result": {
                                "success": success,
                                "contentItems": [
                                    {"type": "inputText", "text": content}
                                ],
                            },
                        }
                    )
                elif (
                    same
                    and method == "item/completed"
                    and params["item"]["type"] == "agentMessage"
                ):
                    completed.append(params["item"]["text"])
                elif same and method == "item/agentMessage/delta":
                    parts.append(params["delta"])
                elif (
                    method == "turn/completed"
                    and params.get("threadId") == self.thread
                    and params["turn"]["id"] == turn
                ):
                    if params["turn"]["status"] != "completed":
                        raise HillclimbError(
                            f"Temper turn {params['turn']['status']} · {params['turn'].get('error') or 'no detail'}"
                        )
                    return "\n".join(completed) or "".join(parts) or "Turn completed"
            raise HillclimbError("Temper disconnected during turn")

    async def close(self):
        if self.backend is not None:
            await self.backend.close()
        self.backend = self.thread = None
