"""Route table tests (GEN-008)."""

import os

import pytest

from oasis_generation.routes import HotRouteTable, NoRoute, RouteConfigError, RouteTable, Target

BASE = {
    "pools": {
        "vllm-eu": {"base_url": "http://vllm:8000/v1/", "region": "eu"},
        "scw-l40s": {"base_url": "http://scw:8000/v1", "region": "eu-fr"},
        "us-pool": {"base_url": "http://us:8000/v1", "region": "us"},
        "undeclared": {"base_url": "http://x:8000/v1"},
    },
    "bedrock": {"default": {"region": "us-east-1"}, "eu": {"region": "eu-central-1"}},
    "remotes": {"vm": {"base_url": "https://gw.example/v1", "region": "eu"}},
    "tenants": {
        "acme": {"residency": "eu", "fallback": "allowed"},
        "strict": {"residency": "eu", "fallback": "never"},
        "*": {"residency": "any", "fallback": "allowed"},
    },
    "routes": [
        {"tenant": "*", "model": "gemma", "targets": ["own:us-pool", "own:vllm-eu", "bedrock:default"]},
        {"tenant": "acme", "model": "*", "targets": ["bedrock:eu", "own:vllm-eu"]},
        {"tenant": "*", "model": "*", "targets": ["bedrock:default"]},
    ],
}


def tbl(**over):
    d = {**BASE, **over}
    return RouteTable.from_dict(d)


def names(r):
    return [str(t) for t in r.targets]


def test_exact_beats_wildcard_and_order_kept():
    assert names(tbl().resolve("someone", "gemma")) == ["own:us-pool", "own:vllm-eu", "bedrock:default"]


def test_tenant_wildcard_model():
    assert names(tbl().resolve("acme", "anything")) == ["bedrock:eu", "own:vllm-eu"]


def test_global_default():
    assert names(tbl().resolve("someone", "other")) == ["bedrock:default"]


def test_eu_residency_drops_non_eu_never_falls_back():
    # acme asks for gemma: the "*"/gemma rule matches (exact model beats tenant wildcard? no:
    # (tenant, model) then (tenant,"*") -> acme/* wins), so check the drop explicitly:
    t = tbl(routes=[{"tenant": "acme", "model": "m", "targets": ["own:us-pool", "bedrock:default", "own:scw-l40s"]}])
    assert names(t.resolve("acme", "m")) == ["own:scw-l40s"]


def test_eu_with_nothing_left_raises_not_leaks():
    t = tbl(routes=[{"tenant": "acme", "model": "m", "targets": ["own:us-pool", "bedrock:default"]}])
    with pytest.raises(NoRoute):
        t.resolve("acme", "m")


def test_undeclared_region_is_not_eu():
    t = tbl(routes=[{"tenant": "acme", "model": "m", "targets": ["own:undeclared"]}])
    with pytest.raises(NoRoute):
        t.resolve("acme", "m")


def test_fallback_never_keeps_first_survivor_only():
    t = tbl(routes=[{"tenant": "strict", "model": "m", "targets": ["own:us-pool", "own:vllm-eu", "own:scw-l40s"]}])
    # us-pool dropped by residency FIRST, then fallback=never keeps the first survivor.
    assert names(t.resolve("strict", "m")) == ["own:vllm-eu"]


def test_fallback_never_any_residency():
    t = tbl(
        tenants={"x": {"residency": "any", "fallback": "never"}},
        routes=[{"tenant": "x", "model": "m", "targets": ["own:us-pool", "own:vllm-eu"]}],
    )
    assert names(t.resolve("x", "m")) == ["own:us-pool"]


def test_unknown_tenant_uses_star_limits():
    assert tbl().limits_for("nobody").residency == "any"


def test_byok_region_from_registry():
    t = RouteTable.from_dict(
        {**BASE, "routes": [{"tenant": "acme", "model": "m", "targets": ["byok:eu-co", "byok:us-co"]}]},
        byok_regions={"eu-co": "eu-west-1", "us-co": "us-east-1"},
        byok_known={"eu-co", "us-co"},
    )
    assert names(t.resolve("acme", "m")) == ["byok:eu-co"]


def test_per_pool_base_url():
    t = tbl()
    assert t.endpoint(Target.parse("own:vllm-eu")).base_url == "http://vllm:8000/v1"  # trailing / stripped
    assert t.endpoint(Target.parse("own:scw-l40s")).base_url == "http://scw:8000/v1"


@pytest.mark.parametrize("raw", ["", "own", "own:", "nope:x", "own:a:b", ":x"])
def test_bad_target_syntax(raw):
    with pytest.raises(RouteConfigError):
        Target.parse(raw)


def test_unknown_reference_rejected():
    with pytest.raises(RouteConfigError):
        tbl(routes=[{"tenant": "*", "model": "m", "targets": ["own:ghost"]}])
    with pytest.raises(RouteConfigError):
        tbl(routes=[{"tenant": "*", "model": "m", "targets": ["bedrock:ghost"]}])


def test_unknown_key_rejected():
    with pytest.raises(RouteConfigError):
        RouteTable.from_dict({"poools": {}})


def test_hot_reload_picks_up_edit_and_survives_bad_edit(tmp_path=None):
    import tempfile, yaml, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "routes.yaml"
    p.write_text(yaml.safe_dump(BASE))
    h = HotRouteTable(p)
    assert names(h.current().resolve("someone", "other")) == ["bedrock:default"]

    edited = {**BASE, "routes": [{"tenant": "*", "model": "*", "targets": ["own:vllm-eu"]}]}
    p.write_text(yaml.safe_dump(edited))
    os.utime(p, ns=(1, 99_000_000_000))  # force a distinct mtime
    assert names(h.current().resolve("someone", "other")) == ["own:vllm-eu"]
    assert h.last_error is None

    p.write_text("routes: [ {tenant: '*', targets: [own:ghost]} ")  # broken yaml
    os.utime(p, ns=(2, 98_000_000_000))
    assert names(h.current().resolve("someone", "other")) == ["own:vllm-eu"]  # kept old
    assert h.last_error

    p.write_text(yaml.safe_dump({**BASE, "routes": [{"tenant": "*", "model": "*", "targets": ["own:ghost"]}]}))
    os.utime(p, ns=(3, 97_000_000_000))
    assert names(h.current().resolve("someone", "other")) == ["own:vllm-eu"]  # semantic error also kept
    assert "ghost" in h.last_error


def test_initial_bad_file_raises():
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "r.yaml"
    p.write_text("pools: 5")
    with pytest.raises(RouteConfigError):
        HotRouteTable(p)
