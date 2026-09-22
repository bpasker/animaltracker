"""Loading a model and running a forward pass never overlap (BUGS.md item 5).

``torch.load`` of SpeciesNet's classifier unpickles a ``torch.fx``
GraphModule; torch rebuilds it by tracing, and the tracer patches
``torch.nn.Module.__call__`` for the whole process while it runs. A
MegaDetector forward on another thread in that window dies inside the tracer
with ``NameError: module is not installed as a submodule``. On production
that was the only error in the journal: 8 tracebacks in 7 days, one lost
frame each, on the first clip after every restart.

The race is real and reproducible with plain torch (no models): a thread
running a small module's forward while another thread symbolically traces an
unrelated module raises exactly that NameError. ``MODEL_LOAD_GATE`` closes
it: forwards hold the gate shared, a load holds it alone.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from animaltracker import detector as detector_mod
from animaltracker.detector import MODEL_LOAD_GATE, BaseDetector, ModelLoadGate

SOURCE = (Path(__file__).resolve().parents[1] / "src" / "animaltracker" / "detector.py").read_text()


def run_threads(n, target):
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads), "a thread never finished: deadlock"


# --- the gate on its own ---------------------------------------------------------------------

def test_forwards_overlap_each_other_but_never_a_load():
    gate = ModelLoadGate()
    peak_forwards = [0]
    seen_loading_during_forward = []
    seen_forwards_during_load = []
    lock = threading.Lock()

    def forward(i):
        for _ in range(200):
            gate.begin_forward()
            try:
                with lock:
                    peak_forwards[0] = max(peak_forwards[0], gate.forwards_in_flight)
                if gate.loading_now:
                    seen_loading_during_forward.append(i)
                time.sleep(0.0002)
            finally:
                gate.end_forward()

    def load(i):
        for _ in range(20):
            with gate.loading():
                if gate.forwards_in_flight:
                    seen_forwards_during_load.append(gate.forwards_in_flight)
                time.sleep(0.001)

    run_threads(6, lambda i: load(i) if i >= 4 else forward(i))

    assert peak_forwards[0] > 1, "forwards must be able to overlap, or inference serialises across backends"
    assert seen_loading_during_forward == []
    assert seen_forwards_during_load == []


def test_a_load_is_not_starved_by_back_to_back_forwards():
    """Three cameras taking turns keep the shared count above zero almost
    always; a waiting load must still get in (new forwards wait for it)."""
    gate = ModelLoadGate()
    stop = threading.Event()
    got_in = threading.Event()

    def forward(i):
        while not stop.is_set():
            gate.begin_forward()
            try:
                time.sleep(0.002)
            finally:
                gate.end_forward()

    def load(i):
        with gate.loading():
            got_in.set()

    workers = [threading.Thread(target=forward, args=(i,)) for i in range(3)]
    for w in workers:
        w.start()
    time.sleep(0.02)
    loader = threading.Thread(target=load, args=(0,))
    loader.start()
    loader.join(timeout=5)
    stop.set()
    for w in workers:
        w.join(timeout=5)

    assert got_in.is_set(), "the load never got past the forwards"


def test_a_load_that_raises_releases_the_gate():
    gate = ModelLoadGate()
    with pytest.raises(RuntimeError):
        with gate.loading():
            raise RuntimeError("weights missing")
    assert not gate.loading_now
    gate.begin_forward()          # would block for ever if the gate were still held
    gate.end_forward()


# --- model_lock takes the gate ---------------------------------------------------------------

class Fake(BaseDetector):
    """A backend whose forward is whatever callable it is given."""

    def __init__(self, forward):
        self._forward = forward

    def infer(self, frame, *args, **kwargs):
        with self.model_lock:
            return self._forward()

    @property
    def backend_name(self):
        return "fake"


def test_model_lock_holds_the_gate_shared_and_the_instance_lock():
    inside = []
    det = Fake(lambda: inside.append((MODEL_LOAD_GATE.forwards_in_flight, det._instance_lock().locked())))

    det.infer(np.zeros((2, 2, 3), np.uint8))

    assert inside == [(1, True)]
    assert MODEL_LOAD_GATE.forwards_in_flight == 0 and not det._instance_lock().locked()


def test_the_guard_is_safe_for_concurrent_entries():
    """One guard per detector, entered by every camera thread at once: the
    counter must come back to zero, or a later load waits for ever."""
    det = Fake(lambda: time.sleep(0.001))

    run_threads(8, lambda i: [det.infer(None) for _ in range(50)])

    assert MODEL_LOAD_GATE.forwards_in_flight == 0
    with MODEL_LOAD_GATE.loading():      # must not block
        pass


def test_a_forward_that_raises_releases_both():
    def boom():
        raise ValueError("bad frame")
    det = Fake(boom)
    with pytest.raises(ValueError):
        det.infer(None)
    assert MODEL_LOAD_GATE.forwards_in_flight == 0 and not det._instance_lock().locked()


# --- the real thing: torch's tracer against a forward pass -----------------------------------

torch = pytest.importorskip("torch")


class Big(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(120)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_a_forward_under_the_gate_survives_a_concurrent_fx_trace():
    """The production failure, with the gate in the way. Every forward runs
    through model_lock; every trace runs under loading(). The invariants
    are checked on every iteration, so this is not a timing test: a forward
    never sees a load in progress, and no forward raises."""
    import torch.fx

    victim = Big()
    x = torch.zeros(1, 8)
    errors = []
    forwards = [0]
    overlaps = []
    stop = threading.Event()

    def run_victim():
        if MODEL_LOAD_GATE.loading_now:
            overlaps.append("forward while loading")
        victim(x)

    det = Fake(run_victim)

    def hammer(i):
        while not stop.is_set():
            try:
                det.infer(None)
                forwards[0] += 1
            except Exception as err:  # noqa: BLE001 - that is the bug
                errors.append(repr(err)[:100])

    def trace(i):
        for _ in range(15):
            with MODEL_LOAD_GATE.loading():
                if MODEL_LOAD_GATE.forwards_in_flight:
                    overlaps.append("load while forwards in flight")
                torch.fx.symbolic_trace(Big())
        stop.set()

    run_threads(3, lambda i: trace(i) if i == 2 else hammer(i))

    assert forwards[0] > 0
    assert overlaps == []
    assert errors == [], errors[:2]


# --- every model build takes it ----------------------------------------------------------------

@pytest.mark.parametrize("build", [
    "self.model = YOLO(model_path)",
    "self._detector = _SNDetector(model_name)",
    "self._model = SpeciesNet(model_name)",
])
def test_each_backend_builds_its_model_under_the_gate(build):
    idx = SOURCE.index(build)
    preceding = SOURCE[max(0, idx - 200):idx]
    assert re.search(r"with MODEL_LOAD_GATE\.loading\(\):\s*(#[^\n]*)?\n\s*$", preceding), build


def test_the_download_happens_before_the_gate_not_under_it():
    for name in ("_SNDetector(model_name)", "SpeciesNet(model_name)"):
        idx = SOURCE.index(name)
        before = SOURCE[max(0, idx - 400):idx]
        assert before.index("_resolve_model_files(model_name)") < before.index("MODEL_LOAD_GATE.loading()"), name


def test_every_forward_site_goes_through_model_lock():
    """No backend takes the bare instance lock for a forward pass."""
    assert "with self._instance_lock()" not in SOURCE
    assert SOURCE.count("with self.model_lock") >= 5
