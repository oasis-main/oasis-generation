# dev environment — GEN-001 spike target.
#
# PREREQ (operator, one-time): Scaleway account under hello@oasis-x.io,
# an API key scoped to a dedicated "oasis-generation" project, and
# SCW_ACCESS_KEY / SCW_SECRET_KEY / SCW_DEFAULT_PROJECT_ID in the env
# (never committed). Then flip instance_count to 1.
#
# WHO APPLIES (2026-09-21, Mike): Yes Man, through the CLAW-116 broker
# (oasis-claw/.swarm/INFRA_BROKER.md). He holds the Scaleway key; other bots
# commit a change here and mail him an INFRA-REQUEST. He works in a clean
# `git archive` export of the named commit, so the state cannot live in this
# directory — it would vanish with the export. The local backend below takes
# its path at init time:
#   Yes Man:  terraform init -input=false -backend-config=yesman.backend.hcl
#   Mac host: terraform init -backend-config="path=$HOME/Library/Application Support/oasis-x/tfstate/oasis-generation/dev.tfstate"
# Both paths are the SAME host file (Yes Man mounts that folder at
# /reach/tfstate). The local backend's lock does not cross the container
# boundary, so never run a plan on the Mac while Yes Man applies.
#
# The provider version is pinned EXACTLY: Yes Man has no egress to the
# Terraform registry and installs providers only from the read-only mirror
# Mike stages with oasis-claw/scripts/claw-infra-tools. A version bump here
# needs a matching re-stage there.

terraform {
  backend "local" {}

  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "2.83.1"
    }
  }
}

provider "scaleway" {
  region = "fr-par"
}

variable "instance_count" {
  type    = number
  default = 0 # flip to 1 once the hello@ account + API key exist
}

module "spike" {
  count  = var.instance_count
  source = "../../modules/generation-instance"

  name              = "gen-spike-s"
  tier              = "S"
  instance_type     = "L40S-1-48G"
  state             = "started"
  weights_volume_gb = 60
  served_model_name = "gemma-4-12b-it"
}

output "spike_ip" {
  value = var.instance_count > 0 ? module.spike[0].public_ip : null
}
