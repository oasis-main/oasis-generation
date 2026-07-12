variable "name" {
  type        = string
  description = "Instance name, e.g. gen-<customer>-<tier>"
}

variable "state" {
  type        = string
  default     = "stopped"
  description = "Lifecycle state: started | stopped | standby. The wake controller flips this (or calls the instance API directly)."
  validation {
    condition     = contains(["started", "stopped", "standby"], var.state)
    error_message = "state must be started, stopped, or standby"
  }
}

variable "instance_type" {
  type        = string
  default     = "L40S-1-48G"
  description = "Scaleway GPU instance type; only provision what the requested model needs (plan §2)."
}

variable "tier" {
  type        = string
  default     = "S"
  description = "Model tier (S/M/L/XL/XXL) — tag only."
}

variable "zone" {
  type    = string
  default = "fr-par-2" # effectively the only GPU zone (some L40S in pl-waw-2)
}

variable "image" {
  type        = string
  default     = "ubuntu_jammy_gpu_os_12"
  description = "GPU OS image slug — verify current slug during the spike."
}

variable "weights_volume_gb" {
  type        = number
  default     = 60
  description = "Persistent block volume for model weights + LoRA adapters (30 GB tier S baseline; size to model)."
}

variable "root_volume_gb" {
  type    = number
  default = 40
}

variable "served_model_name" {
  type    = string
  default = "gemma-4-12b-it"
}

variable "runner_image" {
  type        = string
  default     = "ghcr.io/oasis-main/oasis-generation-runner:latest"
  description = "Runner container image (docker/Dockerfile.runner) — weights are NOT baked in."
}

variable "extra_tags" {
  type    = list(string)
  default = []
}
