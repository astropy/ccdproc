# Licensed under a 3-clause BSD style license - see LICENSE.rst

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident

from ccdproc import conftest as escape_log
from ccdproc.tests import _escape_triage


def test_escape_wrapper_guard_is_thread_local(monkeypatch):
    first_entered = Event()
    release_first = Event()
    inspected = []
    logged = []
    original_calls = []

    def original(value):
        original_calls.append(value)
        return value

    guard = escape_log._ReentrancyGuard()
    wrapper = escape_log._make_escape_logging_wrapper(original, "numpy.asarray", guard)
    nested_wrapper = escape_log._make_escape_logging_wrapper(
        original, "numpy.asanyarray", guard
    )

    def foreign_namespace(value):
        inspected.append(value)
        if value == "first":
            assert nested_wrapper("nested") == "nested"
            first_entered.set()
            assert release_first.wait(timeout=5)
        return "foreign"

    def record_escape(_frame, funcname):
        logged.append(funcname)

    monkeypatch.setattr(escape_log, "_foreign_namespace", foreign_namespace)
    monkeypatch.setattr(escape_log, "_escape_site_frame", lambda: None)
    monkeypatch.setattr(escape_log, "record_escape_log", record_escape)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(wrapper, "first")
        try:
            assert first_entered.wait(timeout=5)
            second = executor.submit(wrapper, "second")
            assert second.result(timeout=5) == "second"
        finally:
            release_first.set()
        assert first.result(timeout=5) == "first"

    assert inspected == ["first", "second"]
    assert logged == ["numpy.asarray", "numpy.asarray"]
    assert original_calls == ["nested", "second", "first"]


def test_escape_log_tally_increment_is_serialized(monkeypatch):
    class TrackingLock:
        def __init__(self):
            self.lock = Lock()
            self.owner = None

        def __enter__(self):
            self.lock.acquire()
            self.owner = get_ident()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.owner = None
            self.lock.release()

        def held_by_current_thread(self):
            return self.owner == get_ident()

    class GuardedCounts(dict):
        def __init__(self, lock):
            super().__init__()
            self.lock = lock

        def __getitem__(self, key):
            assert self.lock.held_by_current_thread()
            return self.get(key, 0)

        def __setitem__(self, key, value):
            assert self.lock.held_by_current_thread()
            super().__setitem__(key, value)

    lock = TrackingLock()
    counts = GuardedCounts(lock)
    monkeypatch.setattr(_escape_triage, "_ESCAPE_LOG_COUNTS", counts)
    monkeypatch.setattr(_escape_triage, "_ESCAPE_LOG_COUNTS_LOCK", lock, raising=False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_escape_triage.record_escape_log, None, "numpy.asarray")
            for _ in range(20)
        ]
        for future in futures:
            future.result(timeout=5)

    key = ("<unknown location>", 0, "", "numpy.asarray")
    assert counts.get(key) == 20
