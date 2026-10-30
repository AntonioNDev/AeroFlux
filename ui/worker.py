import itertools
import queue
import threading
import traceback
from typing import NamedTuple


class JobCancelled(Exception):
    pass


class Event(NamedTuple):
    type: str      # progress | done | error | cancelled
    kind: str
    ticket: int
    payload: object


class JobContext:
    def __init__(self, events, kind, ticket, cancel):
        self._events, self.kind, self.ticket, self._cancel = events, kind, ticket, cancel

    def progress(self, fraction: float, message: str = "") -> None:
        if self._cancel.is_set():
            raise JobCancelled
        self._events.put(Event("progress", self.kind, self.ticket, (fraction, message)))


class JobPool:
    """Two worker threads fed by one queue. Results come back through a second
    queue that the Tk thread drains with after(), so widgets are never touched
    from a worker."""

    def __init__(self, workers: int = 2):
        self._jobs = queue.Queue()
        self._events = queue.Queue()
        self._cancels = {}
        self._tickets = itertools.count(1)
        self._threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    def submit(self, kind, fn, *args, supersede=True) -> int:
        if supersede:
            self.cancel(kind)
        ticket = next(self._tickets)
        cancel = threading.Event()
        self._cancels.setdefault(kind, []).append(cancel)
        self._jobs.put((kind, ticket, cancel, fn, args))
        return ticket

    def cancel(self, kind) -> None:
        for flag in self._cancels.get(kind, []):
            flag.set()
        self._cancels[kind] = []

    def cancel_all(self) -> None:
        for kind in list(self._cancels):
            self.cancel(kind)

    def drain(self) -> list[Event]:
        events, latest = [], {}
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            if ev.type == "progress":
                latest[(ev.kind, ev.ticket)] = ev
            else:
                events.append(ev)
        return list(latest.values()) + events

    def shutdown(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)

    def _loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, ticket, cancel, fn, args = job
            if cancel.is_set():
                self._events.put(Event("cancelled", kind, ticket, None))
                continue
            try:
                value = fn(JobContext(self._events, kind, ticket, cancel), *args)
                self._events.put(Event("done", kind, ticket, value))
            except JobCancelled:
                self._events.put(Event("cancelled", kind, ticket, None))
            except Exception as exc:
                self._events.put(Event("error", kind, ticket, (exc, traceback.format_exc())))
