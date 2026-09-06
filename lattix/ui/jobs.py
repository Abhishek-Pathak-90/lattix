"""Background jobs of the UI server: one worker at a time (engine runs share scratch directories and some
engines are not re-entrant), a log and a progress counter the page polls, cooperative cancel (plus killing
the job's subprocess when it has one) and a watchdog timeout."""
from __future__ import annotations

import secrets
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict = field(default_factory=lambda: {"done": 0, "total": 0, "current": ""})
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: dict | None = None
    meta: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    proc: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def say(self, text: str) -> None:
        with self._lock:
            self.log.append(text)

    def set_progress(self, done: int, total: int, current: str = "") -> None:
        with self._lock:
            self.progress = {"done": done, "total": total, "current": current}

    def to_dict(self, *, with_result: bool = True) -> dict:
        with self._lock:
            d = {"id": self.id, "kind": self.kind, "state": self.state.value, "created": self.created,
                 "started": self.started, "finished": self.finished, "progress": dict(self.progress),
                 "log": list(self.log), "error": self.error, "meta": dict(self.meta)}
            if with_result:
                d["result"] = self.result if self.state == JobState.DONE else None
        return d

    def stopped(self) -> bool:
        return self.cancel.is_set()


class JobManager:
    def __init__(self, *, max_workers: int = 1, default_timeout_s: float = 1800.0) -> None:
        self.jobs: dict[str, Job] = {}
        self._gate = threading.BoundedSemaphore(max_workers)
        self._lock = threading.Lock()
        self.default_timeout_s = default_timeout_s

    def submit(self, kind: str, fn: Callable[[Job], Any], *, timeout_s: float | None = None,
               meta: dict | None = None) -> Job:
        job = Job(secrets.token_urlsafe(6), kind, meta=dict(meta or {}))
        with self._lock:
            self.jobs[job.id] = job
        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        t = threading.Thread(target=self._run, args=(job, fn, timeout), daemon=True, name=f"lattix-job-{job.id}")
        t.start()
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any], timeout: float) -> None:
        with self._gate:
            if job.cancel.is_set():
                job.state, job.finished = JobState.CANCELLED, time.time()
                return
            job.state, job.started = JobState.RUNNING, time.time()
            timer = threading.Timer(timeout, self._timeout, args=(job,)) if timeout and timeout > 0 else None
            if timer is not None:
                timer.daemon = True
                timer.start()
            try:
                result = fn(job)
            except Exception as exc:  # noqa: BLE001 - the job records it, the server reports it
                if job.state == JobState.RUNNING:
                    job.error = {"type": type(exc).__name__, "message": str(exc)[:2000],
                                 "traceback": traceback.format_exc()[-4000:]}
                    job.state = JobState.CANCELLED if job.cancel.is_set() else JobState.FAILED
            else:
                if job.state == JobState.RUNNING:
                    job.result = result
                    job.state = JobState.CANCELLED if job.cancel.is_set() else JobState.DONE
            finally:
                if timer is not None:
                    timer.cancel()
                job.finished = time.time()

    def _timeout(self, job: Job) -> None:
        if job.state == JobState.RUNNING:
            job.state = JobState.TIMEOUT
            job.error = {"type": "Timeout", "message": "the job exceeded its time limit"}
            job.cancel.set()
            self._kill(job)

    @staticmethod
    def _kill(job: Job) -> None:
        proc = job.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def cancel(self, jid: str) -> bool:
        job = self.jobs.get(jid)
        if job is None or job.state not in (JobState.QUEUED, JobState.RUNNING):
            return False
        job.cancel.set()
        if job.state == JobState.QUEUED:
            job.state, job.finished = JobState.CANCELLED, time.time()
        self._kill(job)
        return True

    def shutdown(self, wait: bool = False) -> None:
        for job in list(self.jobs.values()):
            if job.state in (JobState.QUEUED, JobState.RUNNING):
                self.cancel(job.id)
        if wait:
            deadline = time.time() + 5.0
            while time.time() < deadline and any(j.state == JobState.RUNNING for j in self.jobs.values()):
                time.sleep(0.05)
