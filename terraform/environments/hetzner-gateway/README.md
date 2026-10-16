# hetzner-gateway (GEN-008, INFRA-0007)

One shared-vCPU server (cx23 default, <=4 vCPU) in fsn1 behind one firewall.
Inbound 443 from anywhere; 22 only from `ssh_source_cidrs` (default none).
Caddy terminates TLS using TLS-ALPN-01 (HTTP challenge disabled, port 80 closed).
No IAM, no DNS, no secrets. Token: `TF_VAR_hcloud_token`.

Required variables: `gateway_hostname`, `acme_email`, `gateway_image` (digest-pinned).
Gateway env / Bedrock credentials are NOT set here; they belong to the IAM plan.

Unverified (no terraform or hcloud in my container; file was NOT validated):
- hcloud provider pin 1.52.0 must match the mirror Yes Man uses.
- server_type cx23/cx33 must exist in fsn1 (`hcloud server-type list`).
- Run `terraform validate` before planning.

## Secrets and auth (review of 028fd9c)

- The gateway runs with `OASIS_GENERATION_REQUIRE_AUTH=1` (needs origin/dev 1514d19 in the image): no tokens -> refuses to start, 503.
- `/opt/gateway/secrets.env` is NOT written by Terraform or cloud-init. It holds `OASIS_GENERATION_SERVICE_TOKENS=fleet:<token>`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Compose (`env_file ... required: true`) will not start the stack without it. Delivery is Mike's decision (proposal: Mac-side script over SSH, 22 open to the Mac only during the push, then closed by an in-place firewall update); afterwards run `docker compose up -d` in /opt/gateway.
- `caddy_image` and `gateway_image` are both required digests. Usage ledger persists in /opt/gateway/usage.
- Open for Mike: gateway_image registry (proposal: GH Actions, linux/amd64, ghcr.io/oasis-main public), acme_email (proposal mike@oasis-x.io), hostname generation.dev.cloud.oasis-x.io. Before plan: `hcloud server-type list` to confirm cx23 in fsn1.
