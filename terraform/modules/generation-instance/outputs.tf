output "instance_id" {
  value = scaleway_instance_server.runner.id
}

output "public_ip" {
  value = scaleway_instance_ip.public.address
}

output "weights_volume_id" {
  value = scaleway_block_volume.weights.id
}

output "state" {
  value = scaleway_instance_server.runner.state
}
