"""SSRF guard tests (GEN-008). DNS is stubbed; no network."""

import ipaddress
import socket

import pytest

from oasis_generation import ssrf
from oasis_generation.ssrf import SSRFError


def _stub_dns(monkeypatch, mapping):
    def fake(host, port, type=0, **_):
        addrs = mapping[host]
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
            for a in addrs
        ]
    monkeypatch.setattr(ssrf.socket, "getaddrinfo", fake)


@pytest.mark.parametrize("url", [
    "http://example.com/v1",            # not https
    "ftp://example.com/",
    "file:///etc/passwd",
    "https://user:pw@example.com/",     # userinfo
    "https://good.com@evil.com/",       # userinfo trick
    "https:///nohost",
    "",
])
def test_bad_shape_refused(url):
    with pytest.raises(SSRFError):
        ssrf.check_url(url)


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "127.1.2.3", "10.0.0.5", "172.16.9.9", "192.168.1.1",
    "169.254.169.254", "169.254.0.1", "0.0.0.0", "100.64.0.1", "100.127.255.254",
    "198.18.0.1", "224.0.0.1", "240.0.0.1", "255.255.255.255",
    "::1", "::", "fe80::1", "fc00::1", "fd00::1", "fd7a:115c:a1e0::1",
    "::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:10.0.0.1",
    "64:ff9b::7f00:1",        # NAT64 of 127.0.0.1
    "2002:7f00:1::1",         # 6to4 of 127.0.0.1
    "ff02::1",
])
def test_blocked_literal_ips(ip):
    with pytest.raises(SSRFError):
        ssrf.check_url(f"https://[{ip}]/" if ":" in ip else f"https://{ip}/")


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"])
def test_public_literal_ips_allowed(ip):
    ssrf.check_url(f"https://[{ip}]/" if ":" in ip else f"https://{ip}/")


def test_obfuscated_ipv4_forms_refused(monkeypatch):
    # inet_aton-style forms: decimal, hex, octal. getaddrinfo normalises these
    # to dotted quads; stub that behaviour and confirm the check still bites.
    _stub_dns(monkeypatch, {"2130706433": ["127.0.0.1"], "0x7f.1": ["127.0.0.1"],
                            "0177.0.0.1": ["127.0.0.1"]})
    for h in ("2130706433", "0x7f.1", "0177.0.0.1"):
        with pytest.raises(SSRFError):
            ssrf.check_url(f"https://{h}/")


def test_dns_to_private_refused(monkeypatch):
    _stub_dns(monkeypatch, {"evil.example": ["10.1.2.3"]})
    with pytest.raises(SSRFError):
        ssrf.check_url("https://evil.example/v1")


def test_dns_metadata_refused(monkeypatch):
    _stub_dns(monkeypatch, {"meta.example": ["169.254.169.254"]})
    with pytest.raises(SSRFError):
        ssrf.pin("https://meta.example/")


def test_mixed_answer_refused_when_any_bad(monkeypatch):
    # An attacker-controlled name returns one public and one private record.
    _stub_dns(monkeypatch, {"mixed.example": ["8.8.8.8", "192.168.0.10"]})
    with pytest.raises(SSRFError):
        ssrf.pin("https://mixed.example/")
    _stub_dns(monkeypatch, {"mixed2.example": ["192.168.0.10", "8.8.8.8"]})
    with pytest.raises(SSRFError):
        ssrf.pin("https://mixed2.example/")


def test_unresolvable_refused(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("nope")
    monkeypatch.setattr(ssrf.socket, "getaddrinfo", boom)
    with pytest.raises(SSRFError):
        ssrf.pin("https://nonexistent.example/")


def test_public_dns_pins_ip_and_keeps_hostname(monkeypatch):
    _stub_dns(monkeypatch, {"api.example.com": ["93.184.216.34"]})
    p = ssrf.pin("https://api.example.com/v1/chat")
    assert p.ip == "93.184.216.34" and p.host == "api.example.com" and p.port == 443
    args = p.request_args()
    assert args["url"] == "https://93.184.216.34/v1/chat"
    assert args["headers"]["Host"] == "api.example.com"
    assert args["extensions"]["sni_hostname"] == "api.example.com"
    assert args["follow_redirects"] is False


def test_pin_nondefault_port_and_ipv6(monkeypatch):
    _stub_dns(monkeypatch, {"v6.example": ["2606:4700:4700::1111"]})
    p = ssrf.pin("https://v6.example:8443/x")
    assert p.request_args()["url"] == "https://[2606:4700:4700::1111]:8443/x"
    assert p.request_args()["headers"]["Host"] == "v6.example:8443"


def test_scoped_ipv6_refused(monkeypatch):
    _stub_dns(monkeypatch, {"s.example": ["fe80::1%eth0"]})
    with pytest.raises(SSRFError):
        ssrf.pin("https://s.example/")


def test_bad_port_refused():
    with pytest.raises(SSRFError):
        ssrf.check_url("https://example.com:99999/")


def test_check_ip_ipv4_mapped_public_ok():
    ssrf.check_ip(ipaddress.ip_address("::ffff:8.8.8.8"))
