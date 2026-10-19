"""Route table (GEN-008): (tenant, model id) -> ordered targets.

Target syntax:   own:<pool> | bedrock:<profile> | byok:<provider> | remote:<gateway>

  own      a self-hosted OpenAI-compatible pool, each with its OWN base_url
           (replaces the single settings.upstream_base_url).
  bedrock  an AWS Bedrock profile (region + optional role) held by the gateway.
  byok     a tenant-supplied provider from the BYOK registry (byok.py).
  remote   another oasis-generation gateway (the per-bot local<->VM switch is
           one input to this table: it just contributes a remote: target).

Tenant limits are enforced HERE, at resolution time, so no caller can forget them:
  residency: eu|any   "eu" drops every target not declared region=eu. It never
                      falls back to a non-EU target; if nothing is left the
                      request fails (NoRoute) rather than leaving the EU.
  fallback:  allowed|never   "never" keeps only the first surviving target.

A target with no declared region is treated as NOT eu (fail closed).

routes.yaml shape (see docs/routes.example.yaml):

  pools:    {<name>: {base_url: ..., region: eu, api_key_env: ...}}
  bedrock:  {<profile>: {region: eu-central-1}}
  remotes:  {<name>: {base_url: ..., region: eu, api_key_env: ...}}
  tenants:  {<tenant>: {residency: eu, fallback: never}}
  routes:   [{tenant: "*", model: gemma-4-12b-agentic, targets: [own:vllm-main, bedrock:default]}]

Match order: exact tenant + exact model, exact tenant + "*", "*" + exact model,
"*" + "*". First matching rule wins; its ordered targets are then filtered.

Hot reload: `RouteTable.current()` stats the file (cheap) and reloads on mtime or
size change. A reload that fails to parse or validate KEEPS the previous table and
records the error; a bad edit never takes the gateway down.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

TargetKind = Literal["own", "bedrock", "byok", "remote"]
_KINDS = ("own", "bedrock", "byok", "remote")


class NoRoute(LookupError):
    """No usable target for this (tenant, model) under the tenant's limits."""


class RouteConfigError(ValueError):
    pass


class Target(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: TargetKind
    name: str

    @classmethod
    def parse(cls, raw: str) -> "Target":
        kind, sep, name = (raw or "").partition(":")
        if not sep or kind not in _KINDS or not name or ":" in name:
            raise RouteConfigError(f"bad target {raw!r}: expected one of {_KINDS} as kind:name")
        return cls(kind=kind, name=name)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


class Endpoint(BaseModel):
    """A self-hosted pool or a remote gateway."""
    model_config = ConfigDict(extra="forbid")
    base_url: str
    region: str | None = None
    api_key_env: str | None = None  # NAME of an env var; never the key itself

    @field_validator("base_url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must be http(s)")
        return v.rstrip("/")


class BedrockProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str
    role_arn: str | None = None


class TenantLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    residency: Literal["eu", "any"] = "any"
    fallback: Literal["allowed", "never"] = "allowed"


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant: str = "*"
    model: str = "*"
    targets: list[str] = Field(min_length=1)


class RoutesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pools: dict[str, Endpoint] = {}
    bedrock: dict[str, BedrockProfile] = {}
    remotes: dict[str, Endpoint] = {}
    tenants: dict[str, TenantLimits] = {}
    routes: list[Rule] = []


def _is_eu(region: str | None) -> bool:
    r = (region or "").lower()
    return r == "eu" or r.startswith("eu-")


@dataclass(frozen=True)
class Resolved:
    tenant: str
    model: str
    targets: tuple[Target, ...]
    limits: TenantLimits


@dataclass
class RouteTable:
    data: RoutesFile
    # byok provider name -> declared region (filled by the BYOK registry); a
    # provider missing from this map is not eu. Injected so this module has no
    # dependency on byok.py.
    byok_regions: dict[str, str | None] = field(default_factory=dict)
    byok_known: set[str] | None = None  # if set, byok targets must be in it

    # ---- construction / validation ---------------------------------------
    @classmethod
    def from_dict(cls, raw: dict, **kw) -> "RouteTable":
        try:
            data = RoutesFile.model_validate(raw or {})
        except ValidationError as exc:
            raise RouteConfigError(str(exc)) from exc
        t = cls(data, **kw)
        t._validate_refs()
        return t

    def _validate_refs(self) -> None:
        for rule in self.data.routes:
            for raw in rule.targets:
                tg = Target.parse(raw)
                known = {
                    "own": self.data.pools,
                    "bedrock": self.data.bedrock,
                    "remote": self.data.remotes,
                }.get(tg.kind)
                if known is not None and tg.name not in known:
                    raise RouteConfigError(f"rule {rule.tenant}/{rule.model}: unknown {tg}")
                if tg.kind == "byok" and self.byok_known is not None and tg.name not in self.byok_known:
                    raise RouteConfigError(f"rule {rule.tenant}/{rule.model}: unknown {tg}")

    # ---- target facts -----------------------------------------------------
    def region_of(self, tg: Target) -> str | None:
        if tg.kind == "own":
            return self.data.pools[tg.name].region
        if tg.kind == "remote":
            return self.data.remotes[tg.name].region
        if tg.kind == "bedrock":
            return self.data.bedrock[tg.name].region
        return self.byok_regions.get(tg.name)

    def endpoint(self, tg: Target) -> Endpoint:
        """base_url for own:/remote: targets (the per-pool URL)."""
        if tg.kind == "own":
            return self.data.pools[tg.name]
        if tg.kind == "remote":
            return self.data.remotes[tg.name]
        raise RouteConfigError(f"{tg} has no endpoint in the route table")

    # ---- resolution -------------------------------------------------------
    def limits_for(self, tenant: str) -> TenantLimits:
        t = self.data.tenants
        return t.get(tenant) or t.get("*") or TenantLimits()

    def resolve(self, tenant: str, model: str) -> Resolved:
        limits = self.limits_for(tenant)
        rule = self._match(tenant, model)
        if rule is None:
            raise NoRoute(f"no route for tenant={tenant!r} model={model!r}")
        targets = [Target.parse(r) for r in rule.targets]
        if limits.residency == "eu":
            targets = [t for t in targets if _is_eu(self.region_of(t))]
        if limits.fallback == "never":
            targets = targets[:1]
        if not targets:
            raise NoRoute(
                f"tenant={tenant!r} model={model!r}: no target satisfies "
                f"residency={limits.residency} fallback={limits.fallback}"
            )
        return Resolved(tenant, model, tuple(targets), limits)

    def _match(self, tenant: str, model: str) -> Rule | None:
        for t, m in ((tenant, model), (tenant, "*"), ("*", model), ("*", "*")):
            for rule in self.data.routes:
                if rule.tenant == t and rule.model == m:
                    return rule
        return None


class HotRouteTable:
    """File-backed table with mtime hot reload. A bad edit keeps the old table."""

    def __init__(self, path: str | os.PathLike, **table_kw) -> None:
        self.path = Path(path)
        self._kw = table_kw
        self._lock = threading.Lock()
        self._sig: tuple[int, int] | None = None
        self._table: RouteTable | None = None
        self.last_error: str | None = None
        self.reload(force=True)
        if self._table is None:
            raise RouteConfigError(f"initial load of {self.path} failed: {self.last_error}")

    def reload(self, force: bool = False) -> bool:
        with self._lock:
            try:
                st = self.path.stat()
            except OSError as exc:
                self.last_error = f"stat failed: {exc}"
                return False
            sig = (st.st_mtime_ns, st.st_size)
            if not force and sig == self._sig:
                return False
            self._sig = sig  # record even on failure so we do not re-parse a bad file each call
            try:
                raw = yaml.safe_load(self.path.read_text()) or {}
                self._table = RouteTable.from_dict(raw, **self._kw)
                self.last_error = None
                return True
            except (RouteConfigError, yaml.YAMLError, OSError) as exc:
                self.last_error = str(exc)
                return False

    def current(self) -> RouteTable:
        self.reload()
        assert self._table is not None
        return self._table
