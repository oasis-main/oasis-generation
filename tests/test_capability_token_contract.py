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
                   frozenset(v["revoked_jti"]))
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
            "not_yet_valid", "malformed", "bad_tier", "bad_billing_state"} <= reasons


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
