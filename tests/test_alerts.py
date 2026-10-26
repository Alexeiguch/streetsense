from streetsense.alerts import AlertManager
from streetsense.config import AlertConfig
from streetsense.types import Alert, Priority


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def mgr(**kw):
    c = Clock()
    return AlertManager(AlertConfig(**kw), clock=c), c


def test_highest_priority_first():
    m, c = mgr()
    m.submit(Alert("crosswalk", "crossing ahead", Priority.NOTICE))
    m.submit(Alert("vehicle:ahead", "car close ahead", Priority.CRITICAL))
    assert m.pop_ready().key == "vehicle:ahead"


def test_global_gap_blocks_non_critical():
    m, c = mgr(global_min_gap_s=1.5)
    m.submit(Alert("a", "a", Priority.NOTICE))
    assert m.pop_ready().key == "a"
    c.t = 0.5
    m.submit(Alert("b", "b", Priority.WARNING))
    assert m.pop_ready() is None  # too soon
    c.t = 1.6
    m.submit(Alert("b", "b", Priority.WARNING))
    assert m.pop_ready().key == "b"


def test_critical_interrupts_gap():
    m, c = mgr(global_min_gap_s=1.5)
    m.submit(Alert("a", "a", Priority.NOTICE))
    m.pop_ready()
    c.t = 0.2
    m.submit(Alert("vehicle:left", "car", Priority.CRITICAL))
    assert m.pop_ready().key == "vehicle:left"


def test_cooldown_dedupes_same_key():
    m, c = mgr(global_min_gap_s=0, default_cooldown_s=6)
    m.submit(Alert("person:ahead", "person ahead, 3 metres", Priority.WARNING))
    assert m.pop_ready() is not None
    for t in (1, 2, 3, 5.9):
        c.t = t
        m.submit(Alert("person:ahead", "person ahead, 2 metres", Priority.WARNING))
        assert m.pop_ready() is None
    c.t = 6.1
    m.submit(Alert("person:ahead", "person ahead, 2 metres", Priority.WARNING))
    assert m.pop_ready().text == "person ahead, 2 metres"


def test_escalation_bypasses_cooldown():
    m, c = mgr(global_min_gap_s=0)
    m.submit(Alert("light:unknown:red", "traffic light is red", Priority.NOTICE))
    m.pop_ready()
    c.t = 1
    m.submit(Alert("light:unknown:red", "traffic light is red", Priority.WARNING))
    assert m.pop_ready() is not None


def test_newer_text_replaces_pending_same_key():
    m, c = mgr(global_min_gap_s=10)
    m.submit(Alert("x", "first", Priority.NOTICE))
    m.pop_ready()
    c.t = 1
    m.submit(Alert("person:ahead", "person ahead, 3 metres", Priority.WARNING))
    c.t = 1.5
    m.submit(Alert("person:ahead", "person ahead, 2 metres", Priority.WARNING))
    c.t = 10.5
    m.submit(Alert("person:ahead", "person ahead, 2 metres", Priority.WARNING))
    assert m.pop_ready().text == "person ahead, 2 metres"


def test_stale_pending_dropped():
    m, c = mgr(global_min_gap_s=5)
    m.submit(Alert("x", "x", Priority.NOTICE))
    m.pop_ready()
    c.t = 1
    m.submit(Alert("crosswalk", "crossing ahead", Priority.NOTICE))
    c.t = 5.5  # gap passed, but the pending alert is 4.5 s old: no longer true
    assert m.pop_ready() is None


def test_queue_bounded():
    m, c = mgr(max_queue=3, global_min_gap_s=100)
    m.submit(Alert("x", "x", Priority.INFO))
    m.pop_ready()
    for i in range(10):
        m.submit(Alert(f"k{i}", "t", Priority.NOTICE))
    assert len(m.pending) <= 3


def test_no_spam_over_a_long_scene():
    """A person standing 2 m ahead for 30 s at 10 fps is announced at most every cooldown."""
    m, c = mgr(global_min_gap_s=1.5, default_cooldown_s=6)
    said = 0
    for i in range(300):
        c.t = i / 10
        m.submit(Alert("person:ahead", "person ahead, 2 metres", Priority.WARNING))
        said += m.pop_ready() is not None
    assert said == 5  # t = 0, 6, 12, 18, 24
