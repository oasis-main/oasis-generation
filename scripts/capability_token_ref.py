"""Reference verifier + vector generator for the Oasis capability token (ORG-171 step 1).

This file IS the executable form of docs/capability-token/CONTRACT.md. It is
deliberately small and has no network access. Step 2 (src/shared/capability_token.py
in oasis-cloud) and the gateway must produce the same accept/reject decision, with
the same reason code, for every vector in vectors.json.

Test key only. The seed below is public and signs nothing but test vectors.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

ISS = "https://auth.dev.cloud.oasis-x.io"
AUD = "oasis-gen"
TYP = "oasis-cap+jwt"
MAX_LIFETIME_S = 900
LEEWAY_S = 30
TIERS = ("free", "personal", "pro", "max")
BILLING = ("none", "active", "suspended")
KNOWN_SCOPES = ("gen:chat", "gen:models", "gen:decide", "ws:session", "ws:fs", "ws:shell")
REQUIRED = ("iss", "aud", "sub", "tier", "billing_state", "scopes", "jti", "iat", "exp")

TEST_SEED_A = bytes(range(32))
TEST_SEED_B = bytes(range(32, 64))  # a key that is NOT in the JWKS


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


_B64U_RE = re.compile(r"^[A-Za-z0-9_-]*$")


def b64u_dec(s: str) -> bytes:
    """Strict base64url, no padding. One byte string has exactly one text form.

    Refuses padding, the std alphabet (+ /), whitespace, and non-canonical
    trailing bits (the decoded bytes must re-encode to the same text). Raises
    ValueError on any of these; callers map that to `malformed`.
    """
    if not _B64U_RE.match(s):
        raise ValueError("not base64url")
    raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if b64u(raw) != s:
        raise ValueError("non-canonical base64url")
    return raw


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate JSON key")
        d[k] = v
    return d


def _loads_strict(b: bytes):
    return json.loads(b, object_pairs_hook=_no_dup_pairs)


def _pub_raw(seed: bytes) -> bytes:
    from cryptography.hazmat.primitives import serialization as s

    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
        s.Encoding.Raw, s.PublicFormat.Raw
    )


def jwks_for(seeds_by_kid: dict[str, bytes]) -> dict:
    return {
        "keys": [
            {"kty": "OKP", "crv": "Ed25519", "alg": "EdDSA", "use": "sig",
             "kid": kid, "x": b64u(_pub_raw(seed))}
            for kid, seed in seeds_by_kid.items()
        ]
    }


def _json(o) -> bytes:
    return json.dumps(o, separators=(",", ":"), sort_keys=True).encode()


def sign_raw(header_bytes: bytes, payload_bytes: bytes, seed: bytes) -> str:
    """Sign exact bytes (for vectors that need non-canonical JSON)."""
    h, p = b64u(header_bytes), b64u(payload_bytes)
    sig = Ed25519PrivateKey.from_private_bytes(seed).sign(f"{h}.{p}".encode())
    return f"{h}.{p}.{b64u(sig)}"


def sign(header: dict, payload: dict, seed: bytes) -> str:
    h, p = b64u(_json(header)), b64u(_json(payload))
    sig = Ed25519PrivateKey.from_private_bytes(seed).sign(f"{h}.{p}".encode())
    return f"{h}.{p}.{b64u(sig)}"


@dataclass
class Result:
    ok: bool
    reason: str  # "ok" or a stable reject code


def verify(token: str, jwks: dict, now: int, required_scope: str | None = None,
           revoked_jti: frozenset[str] = frozenset(),
           not_before_by_sub: dict[str, int] | None = None) -> Result:
    """Return Result(ok, reason). Never raises on bad input.

    not_before_by_sub: per-user revocation (CONTRACT section 6). A token whose
    `iat` is earlier than the value for its `sub` is refused as `revoked`.
    """
    def no(r: str) -> Result:
        return Result(False, r)

    parts = token.split(".")
    if len(parts) != 3:
        return no("malformed")
    try:
        header = _loads_strict(b64u_dec(parts[0]))
        payload = _loads_strict(b64u_dec(parts[1]))
        sig = b64u_dec(parts[2])
    except Exception:
        return no("malformed")
    if not isinstance(header, dict) or not isinstance(payload, dict):
        return no("malformed")
    # 1. algorithm: exactly EdDSA. Never read the key type from the token.
    if header.get("alg") != "EdDSA":
        return no("bad_alg")
    if header.get("typ") != TYP:
        return no("bad_typ")
    kid = header.get("kid")
    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if not isinstance(kid, str) or key is None:
        return no("unknown_kid")
    if key.get("kty") != "OKP" or key.get("crv") != "Ed25519":
        return no("unknown_kid")
    # 2. signature over the exact signing input bytes
    try:
        Ed25519PublicKey.from_public_bytes(b64u_dec(key["x"])).verify(
            sig, f"{parts[0]}.{parts[1]}".encode())
    except (InvalidSignature, ValueError, KeyError):
        return no("bad_signature")
    # 3. claims
    if any(c not in payload for c in REQUIRED):
        return no("missing_claim")
    if payload["iss"] != ISS:
        return no("bad_iss")
    if payload["aud"] != AUD:  # string only; list audiences are refused
        return no("bad_aud")
    ints = (payload["iat"], payload["exp"])
    if not all(type(v) is int for v in ints):
        return no("malformed")
    iat, exp = ints
    if exp <= iat:
        return no("bad_lifetime")
    if exp - iat > MAX_LIFETIME_S:
        return no("lifetime_too_long")
    if now >= exp + LEEWAY_S:
        return no("expired")
    if iat > now + LEEWAY_S:
        return no("not_yet_valid")
    if payload["tier"] not in TIERS:
        return no("bad_tier")
    if payload["billing_state"] not in BILLING:
        return no("bad_billing_state")
    sc = payload["scopes"]
    if not (isinstance(sc, list) and all(isinstance(s, str) for s in sc)):
        return no("malformed")
    if not isinstance(payload["sub"], str) or not payload["sub"]:
        return no("malformed")
    if not isinstance(payload["jti"], str) or not payload["jti"]:
        return no("malformed")
    ws = payload.get("ws")
    if "ws" in payload and not (isinstance(ws, list) and all(isinstance(w, str) for w in ws)):
        return no("malformed")
    if payload["jti"] in revoked_jti:
        return no("revoked")
    nb = (not_before_by_sub or {}).get(payload["sub"])
    if nb is not None and iat < nb:
        return no("revoked")
    if required_scope is not None and required_scope not in sc:
        return no("missing_scope")
    return Result(True, "ok")


def check_inference_allowed(payload: dict) -> int:
    """HTTP status the gateway returns for a verified token on a metered route.

    ORG-172 rule 2: no inference without a card. 200 = allowed, 402 = payment
    required, 403 = suspended.
    """
    if payload["billing_state"] == "suspended":
        return 403
    if payload["tier"] == "free" or payload["billing_state"] != "active":
        return 402
    ref = payload.get("allowance_ref")
    if not isinstance(ref, str) or not ref:
        return 402
    return 200


# ---- vector generation ---------------------------------------------------

NOW = 1_800_000_000
KID_A = "oasis-cap-test-2026-10-a"


def base_payload(**over) -> dict:
    p = {
        "iss": ISS, "aud": AUD, "sub": "user_test_001", "tier": "personal",
        "billing_state": "active", "scopes": ["gen:chat", "gen:models"],
        "allowance_ref": "allow_test_001", "jti": "jti-0001",
        "iat": NOW, "exp": NOW + 600,
    }
    p.update(over)
    return p


def base_header(**over) -> dict:
    h = {"alg": "EdDSA", "typ": TYP, "kid": KID_A}
    h.update(over)
    return h


def build_vectors() -> dict:
    jwks = jwks_for({KID_A: TEST_SEED_A})
    V: list[dict] = []

    def add(name, token, expect, now=NOW, scope=None, revoked=(), note="", not_before=None):
        V.append({"name": name, "token": token, "now": now,
                  "required_scope": scope, "revoked_jti": list(revoked),
                  "not_before_by_sub": dict(not_before or {}),
                  "expect": expect, "note": note})

    ok = sign(base_header(), base_payload(), TEST_SEED_A)
    add("valid_personal", ok, "ok", scope="gen:chat")
    add("valid_no_scope_required", ok, "ok")
    add("valid_at_leeway_edge_after_exp", ok, "ok", now=NOW + 600 + LEEWAY_S - 1)
    add("valid_iat_in_future_within_leeway",
        sign(base_header(), base_payload(iat=NOW + LEEWAY_S, exp=NOW + LEEWAY_S + 600), TEST_SEED_A),
        "ok")
    add("valid_max_lifetime_900",
        sign(base_header(), base_payload(exp=NOW + 900), TEST_SEED_A), "ok")
    add("valid_with_workspace_ids",
        sign(base_header(), base_payload(scopes=["ws:session"], ws=["ws_a", "ws_b"]), TEST_SEED_A),
        "ok", scope="ws:session", note="ws is optional until ORG-170 starts")
    # rejects
    add("expired", ok, "expired", now=NOW + 600 + LEEWAY_S)
    add("not_yet_valid",
        sign(base_header(), base_payload(iat=NOW + LEEWAY_S + 1, exp=NOW + 700), TEST_SEED_A),
        "not_yet_valid")
    add("lifetime_901",
        sign(base_header(), base_payload(exp=NOW + 901), TEST_SEED_A), "lifetime_too_long")
    add("wrong_aud", sign(base_header(), base_payload(aud="oasis-ai"), TEST_SEED_A), "bad_aud")
    add("aud_as_list", sign(base_header(), base_payload(aud=[AUD]), TEST_SEED_A), "bad_aud")
    add("wrong_iss", sign(base_header(), base_payload(iss="https://evil.example"), TEST_SEED_A), "bad_iss")
    add("unknown_kid", sign(base_header(kid="nope"), base_payload(), TEST_SEED_A), "unknown_kid")
    add("signed_by_other_key", sign(base_header(), base_payload(), TEST_SEED_B), "bad_signature")
    add("missing_scope", ok, "missing_scope", scope="gen:decide")
    add("revoked_jti", ok, "revoked", revoked=["jti-0001"])
    add("bad_tier", sign(base_header(), base_payload(tier="platinum"), TEST_SEED_A), "bad_tier")
    add("bad_billing_state",
        sign(base_header(), base_payload(billing_state="maybe"), TEST_SEED_A), "bad_billing_state")
    for c in ("iss", "aud", "sub", "tier", "billing_state", "scopes", "jti", "iat", "exp"):
        p = base_payload()
        del p[c]
        add(f"missing_{c}", sign(base_header(), p, TEST_SEED_A), "missing_claim")
    add("iat_float", sign(base_header(), base_payload(iat=float(NOW)), TEST_SEED_A), "malformed")
    # algorithm confusion
    h_none = b64u(_json({"alg": "none", "typ": TYP, "kid": KID_A}))
    p_ok = b64u(_json(base_payload()))
    add("alg_none", f"{h_none}.{p_ok}.", "bad_alg")
    add("alg_none_with_sig", f"{h_none}.{p_ok}.{b64u(b'x'*64)}", "bad_alg")
    add("alg_HS256", sign(base_header(alg="HS256"), base_payload(), TEST_SEED_A), "bad_alg")
    add("alg_ES256", sign(base_header(alg="ES256"), base_payload(), TEST_SEED_A), "bad_alg")
    add("bad_typ", sign(base_header(typ="JWT"), base_payload(), TEST_SEED_A), "bad_typ")
    add("exp_equals_iat",
        sign(base_header(), base_payload(exp=NOW), TEST_SEED_A), "bad_lifetime")
    add("exp_before_iat",
        sign(base_header(), base_payload(exp=NOW - 1), TEST_SEED_A), "bad_lifetime")
    # strict encoding: one token, one string form
    sg = ok.split(".")
    add("sig_with_padding", f"{sg[0]}.{sg[1]}.{sg[2]}==", "malformed",
        note="padding is refused; a validly signed token must not have a second string form")
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    last = sg[2][-1]
    alt = alpha[(alpha.index(last) & ~0xF) | ((alpha.index(last) + 1) & 0xF)]
    add("sig_noncanonical_last_char", f"{sg[0]}.{sg[1]}.{sg[2][:-1]}{alt}", "malformed",
        note="only the 4 unused low bits of the last char differ; decoded bytes are identical")
    add("payload_std_alphabet_char", f"{sg[0]}.{sg[1].replace('_', '/').replace('-', '+') + '+'}.{sg[2]}",
        "malformed", note="'+' is not in the base64url alphabet")
    # duplicate JSON keys: a first-wins verifier and a last-wins verifier would disagree
    hdr_dup = ('{"alg":"EdDSA","kid":"nope","kid":"%s","typ":"%s"}' % (KID_A, TYP)).encode()
    add("duplicate_key_in_header", sign_raw(hdr_dup, _json(base_payload()), TEST_SEED_A), "malformed")
    pl = _json(base_payload(tier="max")).decode()
    pl_dup = pl[:-1] + ',"tier":"free"}'
    add("duplicate_key_in_payload", sign_raw(_json(base_header()), pl_dup.encode(), TEST_SEED_A),
        "malformed")
    # ws type
    add("ws_not_a_list", sign(base_header(), base_payload(ws="ws_a"), TEST_SEED_A), "malformed")
    add("ws_list_with_non_string",
        sign(base_header(), base_payload(ws=["ws_a", 7]), TEST_SEED_A), "malformed")
    # per-user revocation (CONTRACT section 6)
    add("revoked_by_sub_not_before", ok, "revoked", not_before={"user_test_001": NOW + 1})
    add("sub_not_before_equal_iat_ok", ok, "ok", not_before={"user_test_001": NOW})
    add("sub_not_before_other_user_ok", ok, "ok", not_before={"someone_else": NOW + 1000})
    # tampering after signing (payload swapped, signature kept)
    hp = ok.split(".")
    evil = b64u(_json(base_payload(tier="max", scopes=["gen:chat", "ws:shell"])))
    add("payload_swapped_keep_sig", f"{hp[0]}.{evil}.{hp[2]}", "bad_signature")
    add("signature_truncated", ".".join(hp[:2] + [hp[2][:-2]]), "bad_signature")
    add("two_segments", ".".join(hp[:2]), "malformed")
    add("garbage", "not-a-token", "malformed")
    add("empty", "", "malformed")

    # billing decisions for verified tokens (gateway 402 rule, ORG-172 rule 2)
    B = []
    for name, over, status in [
        ("free_no_card", dict(tier="free", billing_state="none", scopes=["gen:models"]), 402),
        ("free_but_billing_active", dict(tier="free", billing_state="active"), 402),
        ("personal_active", dict(), 200),
        ("personal_no_card", dict(billing_state="none"), 402),
        ("personal_suspended", dict(billing_state="suspended"), 403),
        ("pro_active", dict(tier="pro"), 200),
        ("max_active", dict(tier="max"), 200),
    ]:
        B.append({"name": name, "payload": base_payload(**over), "expect_status": status})
    p = base_payload()
    del p["allowance_ref"]
    B.append({"name": "paid_without_allowance_ref", "payload": p, "expect_status": 402})
    for nm, val in (("empty_string", ""), ("null", None), ("zero", 0), ("list", ["a"])):
        B.append({"name": f"paid_allowance_ref_{nm}", "payload": base_payload(allowance_ref=val),
                  "expect_status": 402})

    return {
        "contract": "oasis-capability-token/1",
        "warning": "TEST KEY ONLY. The seed is public. Never trust this JWKS anywhere but tests.",
        "iss": ISS, "aud": AUD, "typ": TYP, "leeway_s": LEEWAY_S,
        "max_lifetime_s": MAX_LIFETIME_S,
        "test_seed_a_hex": TEST_SEED_A.hex(),
        "jwks": jwks, "token_vectors": V, "billing_vectors": B,
    }


if __name__ == "__main__":
    import sys

    out = build_vectors()
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/capability-token/vectors.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {path}: {len(out['token_vectors'])} token vectors, {len(out['billing_vectors'])} billing vectors")
