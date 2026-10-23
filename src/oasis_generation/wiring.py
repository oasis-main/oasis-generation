"""Process-wide singletons for the GEN-008 pieces, built lazily from Settings.

Everything is OFF unless the matching setting is non-empty, so a deployment that
sets nothing behaves exactly as before. Cached per file path so tests (and a
settings change) get a fresh instance.
"""

from __future__ import annotations

import yaml

from .byok import ByokRegistry
from .config import Settings
from .routes import HotRouteTable
from .xservice import LabelSet, LabelSets, XService

_routes: dict[str, HotRouteTable] = {}
_byok: dict[str, ByokRegistry] = {}
_x: dict[tuple, XService] = {}


def routes(settings: Settings) -> HotRouteTable | None:
    if not settings.routes_file:
        return None
    if settings.routes_file not in _routes:
        _routes[settings.routes_file] = HotRouteTable(settings.routes_file)
    return _routes[settings.routes_file]


def byok(settings: Settings) -> ByokRegistry:
    """One registry per byok_file ('' = an empty registry). Entries load as pending."""
    key = settings.byok_file
    if key not in _byok:
        ports = {443} | {int(x) for x in settings.byok_extra_ports.split(",") if x.strip().isdigit()}
        reg = ByokRegistry(allowed_ports=ports)
        if key:
            with open(key, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            for name, entry in (raw.get("providers") or {}).items():
                reg.register(name, entry)
        _byok[key] = reg
    return _byok[key]


def xservice(settings: Settings) -> XService:
    key = (settings.decider_url, settings.labelsets_file, settings.x_timeout_s)
    if key not in _x:
        ls = LabelSets()
        if settings.labelsets_file:
            with open(settings.labelsets_file, encoding="utf-8") as fh:
                for item in (yaml.safe_load(fh) or {}).get("labelsets", []):
                    ls.register(LabelSet.model_validate(item))
        _x[key] = XService(settings.decider_url, ls, timeout_s=settings.x_timeout_s)
    return _x[key]


def reset() -> None:
    _routes.clear(); _byok.clear(); _x.clear()
