# One per-customer inference instance: GPU server + persistent weights
# volume. Wake/sleep = flipping var.state between "started"/"stopped" —
# Scaleway bills storage only while stopped (verified 2026-07-12; confirm
# on first invoice during the GEN-001 spike).
#
# SKELETON — written pre-account (GEN-001). Validate resource attributes
# against the scaleway/scaleway provider during the spike before first use.

terraform {
  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = ">= 2.78"
    }
  }
}

resource "scaleway_block_volume" "weights" {
  name       = "${var.name}-weights"
  iops       = 15000
  size_in_gb = var.weights_volume_gb
  zone       = var.zone
}

resource "scaleway_instance_ip" "public" {
  zone = var.zone
}

resource "scaleway_instance_server" "runner" {
  name  = var.name
  type  = var.instance_type # e.g. L40S-1-48G / H100-1-80G / H100-SXM-8-80G
  image = var.image         # GPU OS image with NVIDIA drivers + docker
  zone  = var.zone

  # THE wake/sleep lever. "stopped" releases GPU billing, keeps volumes.
  # NOTE: scratch NVMe on GPU types is ERASED on full stop — weights live
  # on the block volume, never scratch.
  state = var.state

  ip_id = scaleway_instance_ip.public.id

  additional_volume_ids = [scaleway_block_volume.weights.id]

  root_volume {
    size_in_gb  = var.root_volume_gb
    volume_type = "sbs_volume"
  }

  user_data = {
    cloud-init = templatefile("${path.module}/cloud-init.yaml.tftpl", {
      served_name  = var.served_model_name
      runner_image = var.runner_image
    })
  }

  # lease-until=<unix seconds> is read by the infra-reaper (oasis-cloud-admin/
  # infra_reaper, ADM-057), which runs every 15 min in GCP and POWERS OFF any
  # running instance whose lease has expired, or that has no lease 3h after its
  # last state change. Power-off stops compute billing and keeps the weights
  # volume. Set var.lease_until for a planned session; the reaper caps any
  # lease at 72h ahead. The future wake controller (GEN-004) extends it.
  tags = concat(
    ["oasis-generation", "tier:${var.tier}"],
    var.lease_until == "" ? [] : ["lease-until=${var.lease_until}"],
    var.extra_tags,
  )
}
