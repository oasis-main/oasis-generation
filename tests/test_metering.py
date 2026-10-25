"""ORG-171: usage events, the disk spool, the 503 rule, and the MQ bridge.

No network. The publisher is a fake; the paho client is a fake made by an
injected factory, so the bridge logic runs without a broker.
"""
import json
import uuid

from oasis_generation import capability as cap
from oasis_generation import metering as m


class FakePublisher:
    def __init__(self, ok=True):
        self.ok = ok
        self.sent: list[tuple[str, dict, int]] = []

    def publish(self, topic, payload, qos):
        if not self.ok:
            return False
        self.sent.append((topic, json.loads(payload), qos))
        return True


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _event(**over):
    base = dict(sub="u1", jti="j1", allowance_ref="allow_u1_2026-10", request_id="req_1",
                client_request_id="c1", model="claude-sonnet-5", backend="bedrock",
                served_by="bedrock", tokens_in=10, tokens_out=5, cache_read_tokens=0,
                cache_write_tokens=0, status="ok", streamed=True)
    base.update(over)
    return m.build_usage_event(**base)


def test_module_does_not_import_paho_at_import_time():
    assert "mqtt" not in vars(m)
    assert "paho" not in vars(m)


def test_uuid7_is_version_7_and_time_ordered():
    a, b = m.uuid7(), m.uuid7()
    assert uuid.UUID(a).version == 7
    assert a < b or a[:13] == b[:13]  # same millisecond may tie on the time part


def test_usage_event_has_every_field_of_interface_5_3():
    ev = _event()
    assert set(ev) >= {
        "event_id", "ts", "service", "capability", "sub", "jti", "allowance_ref", "request_id",
        "client_request_id", "model", "backend", "served_by", "tokens_in", "tokens_out",
        "cache_read_tokens", "cache_write_tokens", "status", "streamed"}
    assert ev["service"] == "gen" and ev["capability"] == "inference"
    assert ev["ts"].endswith("Z") or ev["ts"].endswith("+00:00")
    assert uuid.UUID(ev["event_id"]).version == 7


def test_emit_publishes_qos1_on_usage_topic(tmp_path):
    pub = FakePublisher()
    meter = m.UsageMeter(pub, str(tmp_path))
    meter.emit(_event())
    assert pub.sent[0][0] == "oasis/system/gen/usage/output" == m.USAGE_TOPIC
    assert pub.sent[0][2] == 1
    assert meter.spool_count() == 0
    assert meter.available()


def test_failed_publish_goes_to_spool_and_flush_resends_in_order(tmp_path):
    pub = FakePublisher(ok=False)
    meter = m.UsageMeter(pub, str(tmp_path))
    meter.emit(_event(request_id="r1"))
    meter.emit(_event(request_id="r2"))
    assert meter.spool_count() == 2
    lines = (tmp_path / m.SPOOL_FILE).read_text().splitlines()
    assert [json.loads(x)["event"]["request_id"] for x in lines] == ["r1", "r2"]
    assert meter.flush() == 0  # still down
    pub.ok = True
    assert meter.flush() == 2
    assert [e["request_id"] for _, e, _ in pub.sent] == ["r1", "r2"]
    assert meter.spool_count() == 0
    assert (tmp_path / m.SPOOL_FILE).read_text() == ""


def test_flush_stops_at_first_failure_and_keeps_the_rest(tmp_path):
    class Flaky(FakePublisher):
        def publish(self, topic, payload, qos):
            if len(self.sent) == 1:
                return False
            return super().publish(topic, payload, qos)

    meter = m.UsageMeter(FakePublisher(ok=False), str(tmp_path))
    for i in range(3):
        meter.emit(_event(request_id=f"r{i}"))
    meter.publisher = Flaky()
    assert meter.flush() == 1
    assert meter.spool_count() == 2


def test_no_publisher_spools(tmp_path):
    meter = m.UsageMeter(None, str(tmp_path))
    meter.emit(_event())
    assert meter.spool_count() == 1


def test_spool_survives_restart(tmp_path):
    clk = Clock()
    m.UsageMeter(None, str(tmp_path), clock=clk).emit(_event())
    clk.t += 601
    again = m.UsageMeter(None, str(tmp_path), clock=clk)
    assert again.spool_count() == 1
    assert not again.available()  # oldest is older than 10 minutes


def test_503_rule_on_count(tmp_path):
    meter = m.UsageMeter(None, str(tmp_path), spool_max=2)
    meter.emit(_event())
    meter.emit(_event())
    assert meter.available()          # holds 2: not MORE than the max
    meter.emit(_event())
    assert not meter.available()


def test_503_rule_on_age(tmp_path):
    clk = Clock()
    meter = m.UsageMeter(None, str(tmp_path), clock=clk)
    meter.emit(_event())
    clk.t += 599
    assert meter.available()
    clk.t += 1
    assert not meter.available()


def test_unwritable_spool_makes_metering_unavailable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    meter = m.UsageMeter(None, str(blocker / "spool"))  # a directory under a file
    meter.emit(_event())
    assert not meter.available()


def test_default_spool_max_is_1000(tmp_path):
    assert m.UsageMeter(None, str(tmp_path)).spool_max == 1000


# ---- the MQ bridge (paho) with a fake client --------------------------------

class FakeMsg:
    def __init__(self, topic, payload):
        self.topic, self.payload = topic, payload


class FakePaho:
    def __init__(self):
        self.username = self.password = None
        self.subs = []
        self.published = []
        self.connected_to = None
        self.rc = 0
        self.on_connect = self.on_message = self.on_disconnect = None

    def username_pw_set(self, username, password):
        self.username, self.password = username, password

    def connect_async(self, host, port, keepalive):
        self.connected_to = (host, port)

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def reconnect_delay_set(self, min_delay, max_delay):
        pass

    def subscribe(self, topic, qos):
        self.subs.append((topic, qos))

    def publish(self, topic, payload, qos):
        self.published.append((topic, payload, qos))

        class Info:
            rc = self.rc
        return Info()


def _bridge(state=None):
    fake = FakePaho()
    state = state or cap.AuthState()
    b = m.MqttBridge(host="mq.example", port=1883, username="mq-service-gen", password="pw",
                     client_id="oasis-gen-gateway", state=state, client_factory=lambda cid: fake)
    return b, fake, state


def test_bridge_logs_in_as_service_and_subscribes_on_connect():
    b, fake, _ = _bridge()
    b.start()
    assert fake.username == "mq-service-gen" and fake.password == "pw"
    assert fake.connected_to == ("mq.example", 1883)
    assert not b.connected
    fake.on_connect(fake, None, {}, 0, None)
    assert b.connected
    assert ("oasis/system/auth/revocation/output", 1) in fake.subs
    assert ("oasis/system/auth/billing/output", 1) in fake.subs


def test_bridge_publish_false_when_not_connected():
    b, fake, _ = _bridge()
    b.start()
    assert b.publish(m.USAGE_TOPIC, b"{}", 1) is False
    fake.on_connect(fake, None, {}, 0, None)
    assert b.publish(m.USAGE_TOPIC, b"{}", 1) is True
    fake.rc = 4
    assert b.publish(m.USAGE_TOPIC, b"{}", 1) is False
    fake.on_disconnect(fake, None, {}, 7, None)
    assert not b.connected


def test_bridge_routes_messages_to_state():
    b, fake, state = _bridge()
    b.start()
    fake.on_message(fake, None, FakeMsg(
        "oasis/system/auth/revocation/output",
        json.dumps({"event_id": "e", "ts": "t", "jti": "j9", "reason": "r"}).encode()))
    fake.on_message(fake, None, FakeMsg(
        "oasis/system/auth/billing/output",
        json.dumps({"event_id": "e", "ts": "2026-10-25T00:00:00Z", "sub": "u7",
                    "state": "blocked", "reason": "spend_cap_reached"}).encode()))
    fake.on_message(fake, None, FakeMsg("oasis/system/auth/billing/output", b"not json"))
    fake.on_message(fake, None, FakeMsg("oasis/system/other/output", b"{}"))
    assert "j9" in state.revoked_jti()
    assert state.is_blocked("u7")


def test_client_id_gets_a_unique_suffix():
    a = m.unique_client_id("oasis-gen-gateway")
    b = m.unique_client_id("oasis-gen-gateway")
    assert a.startswith("oasis-gen-gateway-") and a != b
