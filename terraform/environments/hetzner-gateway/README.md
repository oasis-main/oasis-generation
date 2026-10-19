# hetzner-gateway (GEN-008, INFRA-0007)

One shared-vCPU server (cx23 default, <=4 vCPU) in fsn1 behind one firewall.
Inbound 443 from anywhere; 22 only from `ssh_source_cidrs` (default none).
Caddy terminates TLS using TLS-ALPN-01 (HTTP challenge disabled, port 80 closed).
No IAM, no DNS, no secrets. Token: `TF_VAR_hcloud_token`.

Required variables: `gateway_hostname`, `acme_email`, `gateway_image` (digest-pinned).
Gateway env / Bedrock credentials are NOT set here; they belong to the IAM plan.

Unverified (no terraform or hcloud in my container; file was NOT validated):
- hcloud provider pin 1.70.0 must match the mirror Yes Man uses.
- server_type cx23/cx33 must exist in fsn1 (`hcloud server-type list`).
- Run `terraform validate` before planning.

## Secrets and auth (review of 028fd9c)

- The gateway runs with `OASIS_GENERATION_REQUIRE_AUTH=1` (needs origin/dev 1514d19 in the image): no tokens -> refuses to start, 503.
- `/opt/gateway/secrets.env` is NOT written by Terraform or cloud-init. It holds `OASIS_GENERATION_SERVICE_TOKENS=fleet:<token>`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Compose (`env_file ... required: true`) will not start the stack without it. Delivery is Mike's decision (proposal: Mac-side script over SSH, 22 open to the Mac only during the push, then closed by an in-place firewall update); afterwards run `docker compose up -d` in /opt/gateway.
- `caddy_image` and `gateway_image` are both required digests. Usage ledger persists in /opt/gateway/usage.
- Decided by Mike (relayed by console 2026-10-19/16): `acme_email = mike@oasis-x.io`; `gateway_image = ghcr.io/oasis-main/oasis-generation-gateway@sha256:084fc0fd1662f05bf73122a56c796dcbaa7b3a7b2e1eef4a4a3024532bf35357` (built from oasis-generation dev 71b8366 by gateway-image.yml, linux/amd64, public package, no registry credential needed; runs as uid 10001, so cloud-init creates /opt/gateway/usage owned by 10001:10001). Pin the digest, never :dev; a newer push to dev yields a new digest in the workflow run summary.
- Decided by Mike (console, 2026-10-19): secrets path = the Mac pushes `/opt/gateway/secrets.env` over SSH with oasis-claw `scripts/claw-gen-vm-bootstrap push <ipv4> generation.dev.cloud.oasis-x.io` (writes the file, starts the stack, checks health, 401, 200). Port 22 stays open to the Mac's IPv4 /32 only: Yes Man passes `-var 'ssh_source_cidrs=["100.34.209.244/32"]'` (default stays `[]`; validation rejects IPv6 and 0.0.0.0/0). `ssh_public_key` defaults to the public `oasis-gen-vm` ed25519 key (not a secret). cloud-init does NOT start the stack (it cannot without secrets.env); the bootstrap script does. Nothing secret is in Terraform, state or user_data.
- Build order: Yes Man applies the VM (INFRA-0007) -> outputs gateway_ipv4/ipv6 -> separate dev Route53 plan for generation.dev.cloud.oasis-x.io -> Mac runs the bootstrap push.
- Still open: hostname confirmation. Before plan: `hcloud server-type list` to confirm cx23 in fsn1.
