"""ORG-171 backstop A: the full state snapshot from oasis-auth, the warm-up gate and /health.

Pure state tests (no network). The gate and /health tests reuse the capability
fixture style of test_browser_stream.py.
"""
import datetime as _dt
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from oasis_generation import browser_stream as bs, capability as cap, main
from oasis_generation.config import get_settings

NOW = 1_800_000_000.0


def iso(t):
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat()


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def snap(ts=NOW, blocked=(), rsubs=(), rjti=(), **extra):
    ev = {"event_id": uuid.uuid4().hex, "ts": iso(ts),
          "blocked": [{"sub": s} if isinstance(s, str) else s for s in blocked],
          "revoked_subs": [{"sub": s, "not_before": nb} for s, nb in rsubs],
          "revoked_jti": [{"jti": j} for j in rjti]}
    ev.update(extra)
    return ev


def state(t=NOW):
    return cap.AuthState(clock=Clock(t))


def test_not_ready_before_first_snapshot():
    st = state()
    assert not st.state_ready and st.state_age_s() is None and st.blocked_count() == 0


def test_snapshot_makes_ready_and_blocks_listed_subs():
    st = state()
    assert st.apply_snapshot(snap(blocked=["u1", "u2"]))
    assert st.state_ready and st.is_blocked("u1") and st.is_blocked("u2") and st.blocked_count() == 2
    assert st.state_age_s() == 0


def test_empty_snapshot_is_valid_and_ready():
    st = state()
    assert st.apply_snapshot(snap()) is True and st.state_ready


@pytest.mark.parametrize("bad", [
    None, [], "x", {},                                           # not a snapshot
    {"event_id": "", "ts": iso(NOW)},                            # empty id
    {"event_id": "e", "ts": "nonsense"},                         # bad ts
    {"event_id": "e", "ts": iso(NOW), "blocked": "u1"},          # list expected
    {"event_id": "e", "ts": iso(NOW), "blocked": [{"nosub": 1}]},
    {"event_id": "e", "ts": iso(NOW), "blocked": [{"sub": ""}]},
    {"event_id": "e", "ts": iso(NOW), "revoked_subs": [{"sub": "u"}]},             # not_before required
    {"event_id": "e", "ts": iso(NOW), "revoked_subs": [{"sub": "u", "not_before": "5"}]},
    {"event_id": "e", "ts": iso(NOW), "revoked_jti": [{"x": 1}]},
    {"event_id": "e", "ts": iso(NOW), "part": 2, "of": 1},
    {"event_id": "e", "ts": iso(NOW), "part": 1},                # part without of
    {"event_id": "e", "ts": iso(NOW), "part": True, "of": 1},
    {"event_id": "e", "ts": iso(NOW), "blocked": [{"sub": "u", "ts": iso(NOW + 5)}]},  # entry newer than snapshot
])
def test_malformed_snapshot_changes_nothing(bad):
    st = state()
    st.apply_snapshot(snap(blocked=["keep"]))
    assert st.apply_snapshot(bad) is False
    assert st.is_blocked("keep") and st.blocked_count() == 1


def test_one_bad_entry_refuses_the_whole_message():
    st = state()
    ev = snap(blocked=["good", {"sub": 7}])
    assert st.apply_snapshot(ev) is False
    assert not st.is_blocked("good") and not st.state_ready


def test_future_snapshot_refused():
    st = state()
    assert st.apply_snapshot(snap(ts=NOW + cap.SNAPSHOT_FUTURE_S + 1)) is False
    assert not st.state_ready


def test_old_snapshot_ignored_not_ready_not_error():
    st = state()
    assert st.apply_snapshot(snap(ts=NOW - cap.SNAPSHOT_MAX_AGE_S - 1, blocked=["u1"])) is True
    assert not st.state_ready and not st.is_blocked("u1")


def test_older_than_applied_snapshot_is_skipped():
    st = state()
    st.apply_snapshot(snap(ts=NOW, blocked=["u1"]))
    assert st.apply_snapshot(snap(ts=NOW - 30, blocked=[])) is True
    assert st.is_blocked("u1")


def test_unlisted_blocked_sub_is_healed_when_its_event_is_old():
    clk = Clock()
    st = cap.AuthState(clock=clk)
    st.apply_billing({"event_id": "e", "ts": iso(NOW - 100), "sub": "u1", "state": "blocked", "reason": "r"})
    assert st.is_blocked("u1")
    clk.t = NOW + 1
    st.apply_snapshot(snap(ts=NOW + 1, blocked=[]))
    assert not st.is_blocked("u1")


def test_recent_block_survives_a_snapshot_that_missed_it():
    st = state()
    st.apply_billing({"event_id": "e", "ts": iso(NOW - 2), "sub": "u1", "state": "blocked", "reason": "r"})
    st.apply_snapshot(snap(ts=NOW, blocked=[]))
    assert st.is_blocked("u1")  # within SNAPSHOT_HEAL_GRACE_S


def test_snapshot_does_not_override_a_newer_ok():
    st = state()
    st.apply_billing({"event_id": "e", "ts": iso(NOW - 1), "sub": "u1", "state": "ok", "reason": ""})
    st.apply_snapshot(snap(ts=NOW, blocked=[{"sub": "u1", "ts": iso(NOW - 50)}]))
    assert not st.is_blocked("u1")


def test_snapshot_does_not_unblock_via_an_older_billing_event_later():
    st = state()
    st.apply_snapshot(snap(ts=NOW, blocked=["u1"]))
    st.apply_billing({"event_id": "e", "ts": iso(NOW - 60), "sub": "u1", "state": "ok", "reason": ""})
    assert st.is_blocked("u1")


def test_revocations_are_merged_and_never_removed():
    st = state()
    st.apply_snapshot(snap(rsubs=[("u1", 100)], rjti=["j1"]))
    assert "j1" in st.revoked_jti() and st.not_before_by_sub()["u1"] == 100
    st.apply_snapshot(snap(ts=NOW + 1, rsubs=[("u1", 50)], rjti=[]))   # smaller not_before, no jti
    assert "j1" in st.revoked_jti() and st.not_before_by_sub()["u1"] == 100
    st.apply_snapshot(snap(ts=NOW + 2, rsubs=[("u1", 200)]))
    assert st.not_before_by_sub()["u1"] == 200


def test_split_snapshot_waits_for_every_part():
    st = state()
    eid = uuid.uuid4().hex
    p1 = snap(blocked=["a"], event_id=eid, part=1, of=2)
    p2 = snap(blocked=["b"], event_id=eid, part=2, of=2)
    assert st.apply_snapshot(p1) is True
    assert not st.state_ready and not st.is_blocked("a")
    assert st.apply_snapshot(p2) is True
    assert st.state_ready and st.is_blocked("a") and st.is_blocked("b")


def test_split_parts_in_any_order_and_duplicates():
    st = state()
    eid = uuid.uuid4().hex
    parts = [snap(blocked=[c], event_id=eid, part=i, of=3) for i, c in enumerate("abc", 1)]
    for p in (parts[2], parts[2], parts[0]):
        st.apply_snapshot(p)
    assert not st.state_ready
    st.apply_snapshot(parts[1])
    assert st.state_ready and st.blocked_count() == 3


def test_split_inconsistent_parts_dropped():
    st = state()
    eid = uuid.uuid4().hex
    st.apply_snapshot(snap(blocked=["a"], event_id=eid, part=1, of=2))
    assert st.apply_snapshot(snap(blocked=["b"], event_id=eid, part=2, of=3)) is False
    assert not st.state_ready


def test_split_unhealed_when_a_part_is_missing():
    """Healing only happens from a complete snapshot, so one part alone never unblocks."""
    clk = Clock()
    st = cap.AuthState(clock=clk)
    st.apply_billing({"event_id": "e", "ts": iso(NOW - 100), "sub": "u1", "state": "blocked", "reason": "r"})
    st.apply_snapshot(snap(ts=NOW, event_id="x", part=1, of=2))
    assert st.is_blocked("u1")


def test_stale_pending_parts_expire():
    clk = Clock()
    st = cap.AuthState(clock=clk)
    st.apply_snapshot(snap(blocked=["a"], event_id="x", part=1, of=2))
    clk.t += cap.SNAPSHOT_PENDING_TTL_S + 1
    st.apply_snapshot(snap(ts=clk.t, blocked=["b"], event_id="x", part=2, of=2))
    assert not st.state_ready  # part 1 expired, so the set is incomplete


def test_pending_table_is_bounded():
    st = state()
    for i in range(cap.SNAPSHOT_PENDING_MAX + 5):
        st.apply_snapshot(snap(event_id=f"e{i}", part=1, of=2))
    assert len(st._pending) <= cap.SNAPSHOT_PENDING_MAX


def test_list_size_cap():
    st = state()
    ev = snap(blocked=[f"u{i}" for i in range(cap.SNAPSHOT_MAX_LIST + 1)])
    assert st.apply_snapshot(ev) is False


def test_state_age_grows():
    clk = Clock()
    st = cap.AuthState(clock=clk)
    st.apply_snapshot(snap())
    clk.t += 42
    assert st.state_age_s() == 42


# ---- runtime gate and /health ----

@pytest.fixture
def rt_factory(monkeypatch, tmp_path):
    def make(fail_open="false", warmup="120"):
        monkeypatch.setenv("OASIS_GENERATION_AUTH_MODE", "capability")
        monkeypatch.setenv("CAPABILITY_ISSUER", "https://auth.test.cloud.oasis-x.io")
        monkeypatch.setenv("CAPABILITY_JWKS_URL", "https://auth.test.cloud.oasis-x.io/.well-known/jwks.json")
        monkeypatch.setenv("USAGE_SPOOL_DIR", str(tmp_path / "spool"))
        monkeypatch.setenv("STATE_WARMUP_FAIL_OPEN", fail_open)
        monkeypatch.setenv("STATE_WARMUP_S", warmup)
        monkeypatch.delenv("OASIS_GENERATION_SERVICE_TOKENS", raising=False)
        get_settings.cache_clear()
        clk = Clock(time.time())

        async def fetch(url):
            return {"keys": []}

        rt = bs.CapabilityRuntime(get_settings(), fetch=fetch, clock=clk)
        bs.set_runtime(rt)
        return rt, clk
    yield make
    bs.set_runtime(None)
    get_settings.cache_clear()


def test_bare_runtime_does_not_require_state(rt_factory):
    rt, _ = rt_factory()
    assert rt.state_available() is True


def test_prod_gate_closed_until_snapshot_even_after_warmup(rt_factory):
    rt, clk = rt_factory(fail_open="false")
    rt.state_required = True
    assert rt.state_available() is False
    clk.t += 10_000
    assert rt.state_available() is False     # prod: fail closed forever
    rt.state.apply_snapshot(snap(ts=clk.t))
    assert rt.state_available() is True


def test_dev_gate_opens_after_warmup_only(rt_factory):
    rt, clk = rt_factory(fail_open="true", warmup="120")
    rt.state_required = True
    assert rt.state_available() is False
    clk.t += 119
    assert rt.state_available() is False
    clk.t += 2
    assert rt.state_available() is True


def test_default_is_fail_closed(monkeypatch):
    monkeypatch.delenv("STATE_WARMUP_FAIL_OPEN", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().state_warmup_fail_open is False
    finally:
        get_settings.cache_clear()


async def _jwks_fetch(url):
    from tests.test_browser_stream import JWKS
    return JWKS


def test_chat_stream_503_until_state_then_not(rt_factory):
    from tests.test_browser_stream import auth, body, JWKS, HOST  # noqa: F401
    rt, clk = rt_factory()
    rt.state_required = True

    rt.jwks = cap.JwksCache(rt.jwks.url, fetch=_jwks_fetch)
    c = TestClient(main.app, base_url=f"https://{HOST}")
    r = c.post("/v1/chat/stream", json=body(), headers=auth())
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "state_unavailable"
    assert r.headers["retry-after"] == "5"
    # Once the snapshot is applied the same request gets past the state gate.
    rt.state.apply_snapshot(snap(ts=time.time()))
    rt.clock = lambda: time.time()
    r2 = c.post("/v1/chat/stream", json=body(), headers=auth())
    assert "state_unavailable" not in r2.text


def test_health_is_always_200_and_reports_state(rt_factory):
    rt, clk = rt_factory()
    rt.state_required = True
    c = TestClient(main.app)
    h = c.get("/health")
    assert h.status_code == 200
    j = h.json()
    assert j["status"] == "ok" and j["state_ready"] is False and j["state_age_s"] is None
    rt.state.apply_snapshot(snap(ts=clk.t, blocked=["u1"]))
    clk.t += 7
    j = c.get("/health").json()
    assert j["state_ready"] is True and j["blocked_count"] == 1 and j["state_age_s"] == 7.0
    assert "spool_count" in j and "metering_available" in j


def test_health_without_runtime_is_plain():
    bs.set_runtime(None)
    j = TestClient(main.app).get("/health").json()
    assert j == {"status": "ok", "service": "oasis-generation"}


def test_watch_state_logs_once_per_episode(rt_factory, caplog):
    rt, clk = rt_factory()
    rt.state_required = True
    clk.t += 200
    with caplog.at_level("ERROR"):
        rt.watch_state()
        rt.watch_state()
    assert sum("no state snapshot" in r.message for r in caplog.records) == 1
    caplog.clear()
    rt.state.apply_snapshot(snap(ts=clk.t))
    clk.t += rt.state_stale_s + 1
    with caplog.at_level("ERROR"):
        rt.watch_state()
        rt.watch_state()
    assert sum("snapshot is" in r.message for r in caplog.records) == 1
    rt.state.apply_snapshot(snap(ts=clk.t))
    rt.watch_state()
    assert "stale" not in rt._state_warned


