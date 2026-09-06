from __future__ import annotations

import subprocess
import sys
import time

from lattix.ui.jobs import JobManager, JobState


def _wait(job, states=(JobState.DONE, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT), timeout=5.0):
    t0 = time.time()
    while job.state not in states and time.time() - t0 < timeout:
        time.sleep(0.01)
    return job.state


def test_done_with_log_progress_and_result():
    jm = JobManager()

    def work(job):
        job.say("hello")
        job.set_progress(1, 2, "a")
        return {"x": 1}

    job = jm.submit("test", work)
    assert _wait(job) == JobState.DONE
    d = job.to_dict()
    assert d["result"] == {"x": 1} and d["log"] == ["hello"] and d["progress"]["done"] == 1 and d["finished"]


def test_failed_records_the_exception():
    jm = JobManager()
    job = jm.submit("test", lambda j: 1 / 0)
    assert _wait(job) == JobState.FAILED and job.error["type"] == "ZeroDivisionError"
    assert job.to_dict()["result"] is None


def test_cancel_is_cooperative_and_kills_a_subprocess():
    jm = JobManager()

    def work(job):
        job.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        job.proc.wait()
        return "finished" if not job.stopped() else "stopped"

    job = jm.submit("test", work)
    _wait(job, states=(JobState.RUNNING,))
    time.sleep(0.2)
    assert jm.cancel(job.id) and _wait(job) == JobState.CANCELLED
    assert job.proc.poll() is not None and not jm.cancel(job.id)


def test_timeout_flips_the_state():
    jm = JobManager(default_timeout_s=0.05)

    def work(job):
        time.sleep(0.4)
        return "late"

    job = jm.submit("test", work)
    assert _wait(job, timeout=2.0) == JobState.TIMEOUT and job.error["type"] == "Timeout"
    time.sleep(0.5)
    assert job.state == JobState.TIMEOUT and job.result is None       # the late result is discarded


def test_queue_runs_one_at_a_time_and_shutdown_cancels_queued():
    jm = JobManager()
    order = []

    def slow(job):
        order.append(job.id)
        time.sleep(0.3)
        return job.id

    j1 = jm.submit("a", slow)
    j2 = jm.submit("b", slow)
    time.sleep(0.05)
    assert j1.state == JobState.RUNNING and j2.state == JobState.QUEUED
    jm.shutdown()
    assert _wait(j2) == JobState.CANCELLED and jm.get(j1.id) is j1 and jm.get("nope") is None
