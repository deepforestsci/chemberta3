variable "vpc_remote_state_key" {
  description = "terraform state key of vpc"
  default = "global/vpc/terraform.tfstate"
  type = string
}

variable "ubuntu_2204_ami" {
  description = "ubuntu 22.04 ami id for deep learning"
  default = "ami-0d8efac6374295c8a"
  type = string
}

variable "vpc_remote_state_bucket" {
  description = "terraform vpc remote state bucket"
  default = "dfs-terraform-states"
  type = string
}

variable "dev_key_pair" {
  description = "key pair of dev instances"
  default = "dev-instance"
  type = string
}
