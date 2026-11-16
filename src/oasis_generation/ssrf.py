"""SSRF guard for every outbound call to a customer-supplied (BYOK) endpoint.

GEN-008. A BYOK registry entry carries a tenant-chosen base_url. Without a
guard, a tenant could point it at the cloud metadata service
(169.254.169.254), at a loopback admin port, or at a private-network service,
and use the gateway as a proxy into our network. The rules, all enforced here:

  1. https only. No http, no other scheme.
  2. No userinfo in the URL (https://good.com@evil.internal/ style tricks).
  3. The hostname is RESOLVED, and EVERY returned address is checked. One bad
     address among several refuses the whole call (an attacker controls DNS and
     can return a public and a private A record together).
  4. Refused ranges: loopback, private (RFC 1918 / ULA), link-local (incl.
     169.254.169.254 and fe80::/10), unspecified, multicast, reserved,
     carrier-grade NAT (100.64.0.0/10, where Tailscale and many tunnels live),
     benchmarking (198.18.0.0/15), the Tailscale ULA fd7a:115c:a1e0::/48,
     documentation ranges, and IPv4-mapped / NAT64 / 6to4 / Teredo IPv6 forms
     are unwrapped and their embedded IPv4 checked.
  5. DNS rebinding: checking and then letting the HTTP client resolve AGAIN
     leaves a window in which the answer changes. `pin()` returns the checked
     address so the caller connects to THAT ip (SNI and Host stay the original
     hostname). Use `pin()` for the real call; `check_url()` alone is for
     registration-time validation, never as the only guard on a call.
  6. Port: 443 only unless the OPERATOR allow-lists more (never the tenant);
     port 0 is refused (it must not default to 443).
  7. Redirects must not be followed (a 302 to http://169.254.169.254/ would
     bypass every check above). The caller must pass follow_redirects=False.

This module is stdlib only on purpose: it must stay auditable in one read.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit


class SSRFError(ValueError):
    """The URL or one of its resolved addresses is not allowed."""


_BLOCKED_NETS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",          # "this network"
        "100.64.0.0/10",      # carrier-grade NAT / Tailscale / tunnels
        "192.0.0.0/24",       # IETF protocol assignments
        "192.0.2.0/24",       # TEST-NET-1
        "192.88.99.0/24",     # 6to4 relay anycast (deprecated; Python calls it global)
        "198.18.0.0/15",      # benchmarking
        "198.51.100.0/24",    # TEST-NET-2
        "203.0.113.0/24",     # TEST-NET-3
        "240.0.0.0/4",        # reserved
        "fd7a:115c:a1e0::/48",  # Tailscale ULA
        "2001:db8::/32",      # IPv6 documentation
    )
]

# Prefixes that embed an IPv4 address; the embedded address is what matters.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_6TO4 = ipaddress.ip_network("2002::/16")
_TEREDO = ipaddress.ip_network("2001::/32")


def _embedded_v4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in _6TO4:
        return ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
    if ip in _TEREDO:
        # Teredo stores the client IPv4 XOR 0xFFFFFFFF in the low 32 bits.
        return ipaddress.IPv4Address((int(ip) & 0xFFFFFFFF) ^ 0xFFFFFFFF)
    return None


def check_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Raise SSRFError unless `ip` is an ordinary public unicast address."""
    if isinstance(ip, ipaddress.IPv6Address):
        inner = _embedded_v4(ip)
        if inner is not None:
            check_ip(inner)  # an embedded v4 must pass on its own merits
            # NAT64/6to4/Teredo of a public v4 is still a tunnel form; refuse
            # anyway unless it is plain IPv4-mapped (handled by the recursion
            # above and acceptable only if the v4 is public).
            if ip.ipv4_mapped is None:
                raise SSRFError(f"tunnel-form IPv6 address refused: {ip}")
            return
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
        or not ip.is_global
    ):
        raise SSRFError(f"address not allowed: {ip}")
    for net in _BLOCKED_NETS:
        if ip.version == net.version and ip in net:
            raise SSRFError(f"address in blocked range {net}: {ip}")


def _parse(url: str):
    if not isinstance(url, str) or not url:
        raise SSRFError("empty url")
    parts = urlsplit(url)
    if parts.scheme.lower() != "https":
        raise SSRFError(f"only https is allowed, got {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise SSRFError("userinfo in url is not allowed")
    host = parts.hostname
    if not host:
        raise SSRFError("url has no host")
    try:
        explicit = parts.port
    except ValueError as exc:
        raise SSRFError(f"bad port: {exc}") from exc
    if explicit is not None and not (1 <= explicit <= 65535):
        raise SSRFError(f"bad port: {explicit}")   # port 0 must not fall through to 443
    return parts, host, (443 if explicit is None else explicit)


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFError(f"cannot resolve {host!r}: {exc}") from exc
    addrs = []
    for _fam, _t, _p, _c, sockaddr in infos:
        a = sockaddr[0]
        if "%" in a:  # scoped IPv6 (fe80::1%eth0): link-local, refuse
            raise SSRFError(f"scoped address refused: {a}")
        if a not in addrs:
            addrs.append(a)
    if not addrs:
        raise SSRFError(f"{host!r} resolved to nothing")
    return addrs


@dataclass(frozen=True)
class Pinned:
    """A checked destination. Connect to `ip`, but keep `host` for SNI/Host."""

    url: str          # original URL
    host: str         # original hostname (SNI + Host header)
    port: int
    ip: str           # the address that passed every check
    all_ips: tuple[str, ...]

    def request_args(self) -> dict:
        """kwargs for httpx so the connection goes to the pinned ip while TLS
        verifies against the original hostname. follow_redirects MUST stay off.
        """
        parts = urlsplit(self.url)
        ip_host = f"[{self.ip}]" if ":" in self.ip else self.ip
        netloc = f"{ip_host}:{self.port}" if self.port != 443 else ip_host
        pinned_url = parts._replace(netloc=netloc).geturl()
        return {
            "url": pinned_url,
            "headers": {"Host": parts.netloc},
            "extensions": {"sni_hostname": self.host},
            "follow_redirects": False,
        }


# Only 443 by default: a tenant-chosen port is a port scanner (22, 6379, 9200...).
# Anything else must be allow-listed by the operator, never by the tenant.
DEFAULT_ALLOWED_PORTS: frozenset[int] = frozenset({443})


def pin(url: str, allowed_ports: frozenset[int] | set[int] = DEFAULT_ALLOWED_PORTS) -> Pinned:
    """Validate `url`, resolve it, check EVERY address, return a pinned target."""
    _parts, host, port = _parse(url)
    if port not in allowed_ports:
        raise SSRFError(f"port {port} is not allowed")
    # A literal IP in the URL is checked directly (no DNS).
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        addrs = _resolve(host, port)
    else:
        addrs = [str(literal)]
    for a in addrs:
        check_ip(ipaddress.ip_address(a))
    return Pinned(url=url, host=host, port=port, ip=addrs[0], all_ips=tuple(addrs))


def check_url(url: str, allowed_ports: frozenset[int] | set[int] = DEFAULT_ALLOWED_PORTS) -> None:
    """Registration-time validation. Raises SSRFError; returns None if fine."""
    pin(url, allowed_ports)
