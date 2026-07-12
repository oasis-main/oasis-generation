# dev environment — GEN-001 spike target.
#
# PREREQ (operator, one-time): Scaleway account under hello@oasis-x.io,
# an API key scoped to a dedicated "oasis-generation" project, and
# SCW_ACCESS_KEY / SCW_SECRET_KEY / SCW_DEFAULT_PROJECT_ID in the env
# (never committed). Then flip instance_count to 1 and:
#   terraform init && terraform plan && terraform apply

terraform {
  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = ">= 2.78"
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
