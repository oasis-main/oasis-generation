"""ORG-171 step 1: the capability-token contract, as executable vectors.

Gate 1 (RESEARCH_STANDARD): scramble each claimed input and confirm the
decision moves. A verifier that accepts everything would pass the "valid"
vectors, so the reject vectors and the scramble test are what carry weight.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
cryptography = pytest.importorskip("cryptography")
import capability_token_ref as ref  # noqa: E402

VEC_PATH = ROOT / "docs" / "capability-token" / "vectors.json"
VEC = json.loads(VEC_PATH.read_text())


def test_vectors_file_matches_generator():
    # Ed25519 is deterministic, so the checked-in file must equal a fresh build.
    assert json.loads(json.dumps(ref.build_vectors(), sort_keys=True)) == VEC


@pytest.mark.parametrize("v", VEC["token_vectors"], ids=lambda v: v["name"])
def test_token_vector(v):
    r = ref.verify(v["token"], VEC["jwks"], v["now"], v["required_scope"],
                   frozenset(v["revoked_jti"]), v["not_before_by_sub"])
    assert r.reason == v["expect"]
    assert r.ok == (v["expect"] == "ok")


@pytest.mark.parametrize("v", VEC["billing_vectors"], ids=lambda v: v["name"])
def test_billing_vector(v):
    assert ref.check_inference_allowed(v["payload"]) == v["expect_status"]


def test_vector_set_has_both_outcomes_and_every_reason():
    reasons = {v["expect"] for v in VEC["token_vectors"]}
    assert "ok" in reasons
    assert {"expired", "bad_aud", "bad_iss", "unknown_kid", "bad_signature", "bad_alg",
            "bad_typ", "missing_claim", "missing_scope", "revoked", "lifetime_too_long",
            "not_yet_valid", "malformed", "bad_lifetime", "bad_tier", "bad_billing_state"} <= reasons


@pytest.mark.parametrize("claim,bad", [
    ("iss", "https://evil.example"), ("aud", "other"), ("sub", ""), ("tier", "x"),
    ("billing_state", "x"), ("scopes", "gen:chat"), ("jti", ""), ("iat", "1"), ("exp", "1"),
])
def test_scrambling_each_signed_claim_is_rejected(claim, bad):
    """Re-sign with one claim scrambled: the verifier must reject, not shrug."""
    p = ref.base_payload()
    p[claim] = bad
    tok = ref.sign(ref.base_header(), p, ref.TEST_SEED_A)
    assert not ref.verify(tok, VEC["jwks"], ref.NOW).ok


def test_flipping_any_byte_of_a_valid_token_is_rejected():
    tok = ref.sign(ref.base_header(), ref.base_payload(), ref.TEST_SEED_A)
    assert ref.verify(tok, VEC["jwks"], ref.NOW).ok
    for i in range(len(tok)):
        if tok[i] == ".":
            continue
        flipped = tok[:i] + ("A" if tok[i] != "A" else "B") + tok[i + 1:]
        assert not ref.verify(flipped, VEC["jwks"], ref.NOW).ok, i


def test_no_third_party_jwt_library_needed():
    # The contract must be implementable with the stdlib + one Ed25519 primitive.
    src = (ROOT / "scripts" / "capability_token_ref.py").read_text()
    assert "import jwt" not in src and "jose" not in src


def test_one_token_has_one_string_form():
    """Padding, std-alphabet chars and non-canonical tail bits must all be refused."""
    tok = ref.sign(ref.base_header(), ref.base_payload(), ref.TEST_SEED_A)
    h, p, s = tok.split(".")
    for bad in (f"{h}.{p}.{s}=", f"{h}.{p}.{s}==", f"{h}=.{p}.{s}", f"{h}.{p} .{s}"):
        assert ref.verify(bad, VEC["jwks"], ref.NOW).reason == "malformed", bad
    # every other spelling of the last char's 4 unused bits decodes to the same bytes: refuse
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    i = alpha.index(s[-1])
    variants = [alpha[(i & ~0xF) | k] for k in range(16) if alpha[(i & ~0xF) | k] != s[-1]]
    assert len(variants) == 15
    for c in variants:
        assert ref.verify(f"{h}.{p}.{s[:-1]}{c}", VEC["jwks"], ref.NOW).reason == "malformed"


def test_vectors_carry_the_new_controls():
    names = {v["name"] for v in VEC["token_vectors"]}
    assert {"sig_with_padding", "sig_noncanonical_last_char", "duplicate_key_in_header",
            "duplicate_key_in_payload", "revoked_by_sub_not_before"} <= names
    bnames = {v["name"] for v in VEC["billing_vectors"]}
    assert {"paid_allowance_ref_empty_string", "paid_allowance_ref_null",
            "paid_allowance_ref_zero"} <= bnames


# ---- ORG-170 P1: audience oasis-ws (CONTRACT section 9) --------------------------

WSV = VEC["ws_token_vectors"]


def _ws(v):
    return ref.verify(v["token"], VEC["jwks"], v["now"], v["required_scope"],
                      frozenset(v["revoked_jti"]), v["not_before_by_sub"],
                      audience=v["audience"], session_id=v["session_id"])


@pytest.mark.parametrize("v", WSV, ids=lambda v: v["name"])
def test_ws_token_vector(v):
    r = _ws(v)
    assert r.reason == v["expect"]
    assert r.ok == (v["expect"] == "ok")


def test_ws_vectors_cover_the_new_reasons_and_both_outcomes():
    reasons = {v["expect"] for v in WSV}
    assert {"ok", "bad_aud", "bad_session", "missing_claim", "missing_scope", "malformed",
            "revoked", "expired", "bad_signature", "bad_iss", "lifetime_too_long"} <= reasons
    assert len(WSV) == 29 and len({v["name"] for v in WSV}) == 29


def test_gateway_vectors_unchanged_by_the_ws_audience():
    assert len(VEC["token_vectors"]) == 50 and len(VEC["billing_vectors"]) == 12


def test_gateway_default_audience_is_still_oasis_gen_and_ws_does_not_leak_in():
    tok = ref.sign(ref.base_header(), ref.ws_payload(), ref.TEST_SEED_A)
    assert ref.verify(tok, VEC["jwks"], ref.NOW).reason == "bad_aud"   # ws token at the gateway
    gen = ref.sign(ref.base_header(), ref.base_payload(ws=["sess_a"]), ref.TEST_SEED_A)
    assert ref.verify(gen, VEC["jwks"], ref.NOW).ok                     # unchanged
    assert ref.verify(gen, VEC["jwks"], ref.NOW, audience=ref.AUD_WS, session_id="sess_a").reason == "bad_aud"


def test_ws_session_must_match_exactly_for_every_near_miss():
    tok = ref.sign(ref.base_header(), ref.ws_payload(ws=["sess_a"]), ref.TEST_SEED_A)
    for sid in ("sess_a", ):
        assert ref.verify(tok, VEC["jwks"], ref.NOW, audience=ref.AUD_WS, session_id=sid).ok
    for sid in ("sess_", "sess_a ", " sess_a", "sess_a\x00", "SESS_A", "sess_aa", "", "sess_b"):
        r = ref.verify(tok, VEC["jwks"], ref.NOW, audience=ref.AUD_WS, session_id=sid)
        assert r.reason == "bad_session", repr(sid)


def test_scrambling_the_ws_claim_or_aud_is_rejected():
    for over in ({"ws": ["sess_x"]}, {"ws": []}, {"ws": "sess_a"}, {"aud": "oasis-gen"}, {"aud": ["oasis-ws"]}):
        tok = ref.sign(ref.base_header(), ref.ws_payload(**over), ref.TEST_SEED_A)
        assert not ref.verify(tok, VEC["jwks"], ref.NOW, audience=ref.AUD_WS, session_id="sess_a").ok, over


def test_flipping_any_byte_of_a_ws_token_is_rejected():
    tok = ref.sign(ref.base_header(), ref.ws_payload(), ref.TEST_SEED_A)
    assert ref.verify(tok, VEC["jwks"], ref.NOW, audience=ref.AUD_WS, session_id="sess_a").ok
    for i in range(len(tok)):
        if tok[i] == ".":
            continue
        flipped = tok[:i] + ("A" if tok[i] != "A" else "B") + tok[i + 1:]
        assert not ref.verify(flipped, VEC["jwks"], ref.NOW, audience=ref.AUD_WS, session_id="sess_a").ok, i


def test_tool_scope_table_matches_the_contract_text():
    doc = (ROOT / "docs" / "capability-token" / "CONTRACT.md").read_text()
    assert VEC["ws_tool_scope"] == ref.WS_TOOL_SCOPE
    for tool, scope in ref.WS_TOOL_SCOPE.items():
        assert scope in ref.KNOWN_SCOPES and f"`{tool}`" in doc
    # every shell tool needs ws:shell, no fs tool does, and nothing defaults for an unknown tool
    assert {t for t, s in ref.WS_TOOL_SCOPE.items() if s == "ws:shell"} == {
        "shell_exec", "shell_start", "shell_read", "shell_stop"}
    assert ref.scope_for_tool("fs_read") == "ws:fs" and ref.scope_for_tool("rm_rf") is None
    assert ref.scope_for_tool("") is None and ref.scope_for_tool("FS_READ") is None
