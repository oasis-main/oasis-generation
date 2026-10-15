"""Static checks on the GEX131 Compose project (GEN-008). No docker needed."""
import re
from pathlib import Path

import yaml

D = Path(__file__).resolve().parent.parent / "docker" / "gex131"
C = yaml.safe_load((D / "docker-compose.yml").read_text())
RAW = (D / "docker-compose.yml").read_text()


def test_every_image_is_a_required_variable_never_a_tag():
    for name, svc in C["services"].items():
        img = svc["image"]
        assert re.fullmatch(r"\$\{[A-Z0-9_]+:\?[^}]+\}", img), f"{name}: image must be a required variable, got {img!r}"
        assert "latest" not in img


def test_no_public_port():
    for name, svc in C["services"].items():
        for p in svc.get("ports", []):
            assert str(p).startswith("127.0.0.1:"), f"{name} publishes {p}"
    assert [n for n, s in C["services"].items() if s.get("ports")] == ["gateway"]


def test_only_gateway_has_egress():
    for name, svc in C["services"].items():
        nets = set(svc["networks"])
        assert ("egress" in nets) == (name == "gateway"), name
    assert C["networks"]["internal"]["internal"] is True


def test_models_mounted_read_only_everywhere():
    for name, svc in C["services"].items():
        for v in svc.get("volumes", []):
            if str(v).startswith("models:"):
                assert str(v).endswith(":ro"), name


def test_no_secret_values_in_file():
    assert "POSTGRES_PASSWORD:" not in RAW and "POSTGRES_PASSWORD=" not in RAW
    assert C["services"]["postgres"]["environment"]["POSTGRES_PASSWORD_FILE"].startswith("/run/secrets/")
    for pat in (r"AKIA[0-9A-Z]{12,}", r"sk-[A-Za-z0-9]{16,}", r"hf_[A-Za-z0-9]{16,}"):
        assert not re.search(pat, RAW)


def test_weight_serving_services_are_offline():
    for n in ("vllm", "decider", "embed", "rerank"):
        env = C["services"][n].get("environment", {})
        flat = env if isinstance(env, dict) else dict(e.split("=", 1) for e in env)
        assert flat.get("HF_HUB_OFFLINE") == "1", n


def test_vllm_is_bf16_with_lora():
    cmd = " ".join(C["services"]["vllm"]["command"])
    assert "--dtype=bfloat16" in cmd and "--enable-lora" in cmd


def test_gateway_routes_file_is_valid_and_points_at_real_services():
    from oasis_generation.routes import RouteTable
    t = RouteTable.from_dict(yaml.safe_load((D / "routes.gex131.yaml").read_text()))
    for pool in t.data.pools.values():
        host = pool.base_url.split("//")[1].split(":")[0]
        assert host in C["services"]
    assert "decider" in C["services"]["gateway"]["environment"]["OASIS_GENERATION_DECIDER_URL"]
