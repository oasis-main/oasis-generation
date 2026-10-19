# hetzner-gateway — GEN-008 / INFRA-0007: the oasis-generation gateway VM.
#
# Scope (hard limits, INFRA-0007): ONE shared-vCPU server (<= 4 vCPU) in fsn1,
# ONE firewall, ONE optional ssh key. No IAM, no DNS, no volumes, no floating IPs,
# no load balancers. DNS record and the Bedrock IAM user are SEPARATE plans.
#
# WHO APPLIES: Yes Man, through the broker, from a clean `git archive` export of
# the named commit; state lives outside this directory (local backend, path given
# at init time):
#   terraform init -input=false -backend-config=yesman.backend.hcl
# Nobody applies without Mike approving the saved plan.
#
# SECRETS: none in these files. The Hetzner token arrives as
#   TF_VAR_hcloud_token
# The provider version is pinned EXACTLY (Yes Man has no registry egress and uses
# the read-only provider mirror). If the mirror holds a different hcloud version,
# change the pin here to match it; do not loosen it to a range.

terraform {
  backend "local" {}

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "1.70.0"
    }
  }
}

provider "hcloud" {
  token = var.hcloud_token
}

# ---------------------------------------------------------------- variables --

variable "hcloud_token" {
  type      = string
  sensitive = true
}

variable "name" {
  type    = string
  default = "oasis-gen-gateway"
}

variable "location" {
  type    = string
  default = "fsn1"
  validation {
    condition     = var.location == "fsn1"
    error_message = "INFRA-0007 limits the gateway to fsn1."
  }
}

# Shared-vCPU, <= 4 vCPU. cx23 = 2 vCPU / 4 GB, cx33 = 4 vCPU / 8 GB.
# VERIFY availability in fsn1 with `hcloud server-type list` before planning.
variable "server_type" {
  type    = string
  default = "cx23"
  validation {
    condition     = contains(["cx23", "cx33"], var.server_type)
    error_message = "INFRA-0007: shared vCPU, 4 vCPU or less (cx23 or cx33)."
  }
}

variable "image" {
  type    = string
  default = "ubuntu-24.04"
}

# Addresses allowed to reach port 22, as IPv4 CIDRs (the Mac's address, /32;
# Yes Man passes it with -var at plan time, e.g. ["203.0.113.7/32"]).
# Empty (default) means NO ssh rule at all: only 443 is open. IPv4 only.
variable "ssh_source_cidrs" {
  type    = list(string)
  default = []
  validation {
    condition     = alltrue([for c in var.ssh_source_cidrs : can(cidrhost(c, 0)) && !strcontains(c, ":") && c != "0.0.0.0/0"])
    error_message = "Port 22: IPv4 CIDRs only, never 0.0.0.0/0."
  }
}

# Public key (NOT a secret) for the Mac-side secrets push (GEN-008, scripts/
# claw-gen-vm-bootstrap). The private half stays on the Mac host only.
variable "ssh_public_key" {
  type    = string
  default = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMG289PIccZQ+0MH1Cx3Yhp0UuyTC6fvm7bLzIRlpjVD oasis-gen-vm"
}

# Hostname Caddy serves and gets a certificate for (TLS-ALPN-01, port 443 only).
# The DNS record is a separate plan; until it resolves here, Caddy keeps retrying.
variable "gateway_hostname" {
  type = string
}

variable "acme_email" {
  type = string
}

# Gateway image pinned by digest. Assumed pullable without credentials:
# registry credentials are NOT part of this plan.
variable "gateway_image" {
  type = string
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.gateway_image))
    error_message = "gateway_image must be pinned by digest (@sha256:...)."
  }
}

# Caddy image pinned by digest (plan 11.2: every image by digest).
variable "caddy_image" {
  type = string
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.caddy_image))
    error_message = "caddy_image must be pinned by digest (@sha256:...)."
  }
}

# ---------------------------------------------------------------- resources --

resource "hcloud_ssh_key" "admin" {
  count      = var.ssh_public_key == "" ? 0 : 1
  name       = "${var.name}-admin"
  public_key = var.ssh_public_key
}

resource "hcloud_firewall" "gateway" {
  name = "${var.name}-fw"

  # 443 from anywhere: the only public port (TLS-ALPN-01 needs no port 80).
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # 22 only from the addresses Yes Man names. No rule when the list is empty.
  dynamic "rule" {
    for_each = length(var.ssh_source_cidrs) > 0 ? [1] : []
    content {
      direction  = "in"
      protocol   = "tcp"
      port       = "22"
      source_ips = var.ssh_source_cidrs
    }
  }
}

resource "hcloud_server" "gateway" {
  name         = var.name
  server_type  = var.server_type
  image        = var.image
  location     = var.location
  ssh_keys     = [for k in hcloud_ssh_key.admin : k.id]
  firewall_ids = [hcloud_firewall.gateway.id]

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }

  user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    gateway_hostname = var.gateway_hostname
    acme_email       = var.acme_email
    gateway_image    = var.gateway_image
    caddy_image      = var.caddy_image
  })

  labels = {
    project = "oasis-generation"
    role    = "gateway"
    item    = "gen-008"
  }

  # Changing user_data must not silently rebuild a serving VM; recreate on purpose.
  lifecycle {
    ignore_changes = [user_data]
  }
}

# ------------------------------------------------------------------ outputs --

output "gateway_ipv4" {
  description = "Point the A record here (separate DNS plan)."
  value       = hcloud_server.gateway.ipv4_address
}

output "gateway_ipv6" {
  description = "Point the AAAA record here (separate DNS plan)."
  value       = hcloud_server.gateway.ipv6_address
}

output "server_id" {
  value = hcloud_server.gateway.id
}
