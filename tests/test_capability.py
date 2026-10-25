"""ORG-171: the production capability-token verifier (src/oasis_generation/capability.py).

Runs EVERY vector in docs/capability-token/vectors.json (token and billing) against
the package verifier, plus the JWKS cache and the revocation / blocked-sub state.
No network: the JWKS fetcher is a fake. The package must not import from scripts/.
"""
import asyncio
import json
import pathlib

import pytest

from oasis_generation import capability as cap

ROOT = pathlib.Path(__file__).resolve().parents[1]
VEC = json.loads((ROOT / "docs" / "capability-token" / "vectors.json").read_text())


@pytest.mark.parametrize("v", VEC["token_vectors"], ids=lambda v: v["name"])
def test_token_vector(v):
    r = cap.verify(
        v["token"], VEC["jwks"], v["now"],
        issuer=VEC["iss"], audience=VEC["aud"],
        required_scope=v["required_scope"],
        revoked_jti=frozenset(v["revoked_jti"]),
        not_before_by_sub=v["not_before_by_sub"],
    )
    assert r.reason == v["expect"]
    assert r.ok == (v["expect"] == "ok")
    if r.ok:
        assert r.claims["sub"] == "user_test_001"
    else:
        assert r.claims is None


@pytest.mark.parametrize("v", VEC["billing_vectors"], ids=lambda v: v["name"])
def test_billing_vector(v):
    assert cap.check_inference_allowed(v["payload"]) == v["expect_status"]


def test_vector_counts_match_contract():
    # CONTRACT.md text says 38/8; the file regenerated on review has more. Every
    # vector in the file is run above; this guards against an empty file.
    assert len(VEC["token_vectors"]) >= 38
    assert len(VEC["billing_vectors"]) >= 8


def test_constants_match_vectors():
    assert cap.TYP == VEC["typ"]
    assert cap.LEEWAY_S == VEC["leeway_s"]
    assert cap.MAX_LIFETIME_S == VEC["max_lifetime_s"]
    assert cap.AUD_GEN == VEC["aud"]


def test_issuer_is_configurable():
    v = next(x for x in VEC["token_vectors"] if x["name"] == "valid_personal")
    r = cap.verify(v["token"], VEC["jwks"], v["now"], issuer="https://auth.prod.cloud.oasis-x.io",
                   audience="oasis-gen")
    assert r.reason == "bad_iss"


def test_package_does_not_import_scripts():
    src = (ROOT / "src" / "oasis_generation" / "capability.py").read_text()
    assert "capability_token_ref" not in src
    assert "scripts" not in src.split('"""', 2)[-1]  # outside the docstring


def test_peek_kid():
    v = next(x for x in VEC["token_vectors"] if x["name"] == "valid_personal")
    assert cap.peek_kid(v["token"]) == "oasis-cap-test-2026-10-a"
    assert cap.peek_kid("garbage") is None
    assert cap.peek_kid("") is None


# ---- JWKS cache ---------------------------------------------------------------

class FakeFetch:
    def __init__(self, docs):
        self.docs = list(docs)
        self.calls = 0

    async def __call__(self, url):
        self.calls += 1
        doc = self.docs[min(self.calls - 1, len(self.docs) - 1)]
        if isinstance(doc, Exception):
            raise doc
        return doc


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _jwks(*kids):
    return {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": k, "x": VEC["jwks"]["keys"][0]["x"]}
                     for k in kids]}


def run(coro):
    return asyncio.run(coro)


def test_jwks_cache_ttl_600():
    f, clk = FakeFetch([_jwks("a"), _jwks("a", "b")]), Clock()
    c = cap.JwksCache("https://auth.example/.well-known/jwks.json", fetch=f, clock=clk)
    assert cap.JWKS_TTL_S == 600
    run(c.get("a"))
    clk.t += 599
    run(c.get("a"))
    assert f.calls == 1
    clk.t += 1
    jwks = run(c.get("a"))
    assert f.calls == 2
    assert {k["kid"] for k in jwks["keys"]} == {"a", "b"}


def test_jwks_unknown_kid_refetch_is_rate_limited():
    f, clk = FakeFetch([_jwks("a"), _jwks("a"), _jwks("a", "new")]), Clock()
    c = cap.JwksCache("https://x/jwks", fetch=f, clock=clk)
    run(c.get("a"))
    run(c.get("unknown"))          # one refetch
    assert f.calls == 2
    run(c.get("unknown"))          # inside the rate-limit window: no refetch
    run(c.get("unknown2"))
    assert f.calls == 2
    clk.t += cap.JWKS_UNKNOWN_KID_MIN_INTERVAL_S
    jwks = run(c.get("new"))
    assert f.calls == 3
    assert "new" in {k["kid"] for k in jwks["keys"]}


def test_jwks_fetch_failure_keeps_last_set():
    f, clk = FakeFetch([_jwks("a"), RuntimeError("down")]), Clock()
    c = cap.JwksCache("https://x/jwks", fetch=f, clock=clk)
    run(c.get("a"))
    clk.t += 601
    jwks = run(c.get("a"))
    assert [k["kid"] for k in jwks["keys"]] == ["a"]


def test_jwks_fetch_failure_cold_gives_empty_set():
    c = cap.JwksCache("https://x/jwks", fetch=FakeFetch([RuntimeError("down")]), clock=Clock())
    assert run(c.get("a")) == {"keys": []}


def test_jwks_drops_non_ed25519_keys():
    doc = {"keys": [{"kty": "RSA", "kid": "r", "n": "x", "e": "AQAB"},
                    {"kty": "OKP", "crv": "X25519", "kid": "x", "x": "abc"},
                    {"kty": "OKP", "crv": "Ed25519", "kid": 7, "x": "abc"},
                    {"kty": "OKP", "crv": "Ed25519", "kid": "ok", "x": "abc"}]}
    assert [k["kid"] for k in cap.sanitize_jwks(doc)["keys"]] == ["ok"]
    assert cap.sanitize_jwks("nope") == {"keys": []}


def test_jwks_url_must_be_https_except_loopback():
    cap.check_jwks_url("https://auth.dev.cloud.oasis-x.io/.well-known/jwks.json")
    cap.check_jwks_url("http://localhost:8000/.well-known/jwks.json")
    cap.check_jwks_url("http://127.0.0.1:8000/.well-known/jwks.json")
    with pytest.raises(ValueError):
        cap.check_jwks_url("http://auth.dev.cloud.oasis-x.io/.well-known/jwks.json")
    with pytest.raises(ValueError):
        cap.check_jwks_url("")


# ---- revocation set, per-sub not-before, blocked subs ------------------------

def test_revocation_by_jti_and_expiry():
    clk = Clock(1_800_000_000)
    st = cap.AuthState(clock=clk)
    st.apply_revocation({"event_id": "e1", "ts": "2027-01-15T08:00:00Z", "jti": "j1", "reason": "logout"})
    assert "j1" in st.revoked_jti()
    clk.t += cap.REVOCATION_HOLD_S + 1
    assert "j1" not in st.revoked_jti()


def test_revocation_by_sub_not_before_int_and_default():
    clk = Clock(1_800_000_000)
    st = cap.AuthState(clock=clk)
    st.apply_revocation({"event_id": "e1", "ts": "x", "sub": "u1", "not_before": 1_800_000_100,
                         "reason": "password_change"})
    assert st.not_before_by_sub() == {"u1": 1_800_000_100}
    # A later event moves the time forward; an older one never moves it back.
    st.apply_revocation({"event_id": "e2", "ts": "x", "sub": "u1", "not_before": 1_800_000_050, "reason": "r"})
    assert st.not_before_by_sub() == {"u1": 1_800_000_100}
    # Without not_before, the receive time is used.
    st.apply_revocation({"event_id": "e3", "ts": "x", "sub": "u2", "reason": "r"})
    assert st.not_before_by_sub()["u2"] == 1_800_000_000
    clk.t = 1_800_000_100 + cap.REVOCATION_HOLD_S + 1
    assert "u1" not in st.not_before_by_sub()


def test_revocation_rejects_bad_payloads():
    st = cap.AuthState(clock=Clock())
    for bad in (None, [], "x", {"jti": 5}, {"sub": ""}, {"sub": "u", "not_before": "soon"},
                {"sub": "u", "not_before": True}, {"reason": "nothing to revoke"}):
        assert st.apply_revocation(bad) is False
    assert st.revoked_jti() == frozenset() and st.not_before_by_sub() == {}


def test_billing_blocked_and_ok():
    st = cap.AuthState(clock=Clock())
    assert st.apply_billing({"event_id": "e", "ts": "2026-10-25T10:00:00Z", "sub": "u1",
                             "state": "blocked", "reason": "spend_cap_reached"})
    assert st.is_blocked("u1")
    # An older event (by ts) does not undo a newer one.
    st.apply_billing({"event_id": "e0", "ts": "2026-10-25T09:00:00Z", "sub": "u1", "state": "ok", "reason": ""})
    assert st.is_blocked("u1")
    st.apply_billing({"event_id": "e2", "ts": "2026-10-25T11:00:00Z", "sub": "u1", "state": "ok", "reason": ""})
    assert not st.is_blocked("u1")
    assert st.apply_billing({"sub": "u1", "state": "maybe"}) is False


def test_verify_uses_state():
    v = next(x for x in VEC["token_vectors"] if x["name"] == "valid_personal")
    st = cap.AuthState(clock=Clock(v["now"]))
    st.apply_revocation({"event_id": "e", "ts": "x", "jti": "jti-0001", "reason": "r"})
    r = cap.verify(v["token"], VEC["jwks"], v["now"], issuer=VEC["iss"], audience="oasis-gen",
                   revoked_jti=st.revoked_jti(), not_before_by_sub=st.not_before_by_sub())
    assert r.reason == "revoked"
