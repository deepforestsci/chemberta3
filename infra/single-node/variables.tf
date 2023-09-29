variable "chemberta_key_pair" {
  description = "key pair of dev instances"
  default = "dev-instance"
  type = string
}

variable "ubuntu_2204_ami" {
  description = "ubuntu 22.04 ami id for deep learning"
  default = "ami-0d8efac6374295c8a"
  type = string
}

variable "vpc_id" {
  description = "vpc id"
  # Default VPC Id
  default = "vpc-0ddf915f26d8b4a8f"
  type = string
}

variable "subnet_id" {
  description = "subnet_id"
  # Default Subnet
  default = "subnet-0ba8129be8da95b0a"
  type = string
}
