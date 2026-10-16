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
