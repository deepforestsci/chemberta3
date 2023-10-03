terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.17.0"
    }
  }
  required_version = ">= 1.2.0"

  backend "s3" {
    bucket = "dfs-terraform-states"
    key = "chemberta3/ec2/terraform.tfstate"
    region = "us-east-2"

    dynamodb_table = "tf-state-locks"
    encrypt = "true"
  }
}

provider "aws" {
  profile = "default"
  region  = "us-east-2"
}

data "terraform_remote_state" "vpc" {
  backend = "s3"

  config = {
    bucket = var.vpc_remote_state_bucket
    key = var.vpc_remote_state_key
    region = "us-east-2"
  }
}

resource "aws_security_group" "chemberta_node_sg" {
  name = "mainVpc/ChembertaNodeSg"
  egress {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      description = "all-traffic"
      from_port = 0
      protocol = -1
      to_port = 0
    }
  egress {
    cidr_blocks      = [
        "0.0.0.0/0"
    ]
    description      = "all-tcp"
    from_port        = 0
    ipv6_cidr_blocks = []
    prefix_list_ids  = []
    protocol         = "tcp"
    security_groups  = []
    self             = false
    to_port          = 65535
  }
  egress {
    cidr_blocks      = []
    description      = "all-traffic-ipv6-egress"
    from_port        = 0
    ipv6_cidr_blocks = [
      "::/0",
    ]
    prefix_list_ids  = []
    protocol         = "-1"
    security_groups  = []
    self             = false
    to_port          = 0
  }
  egress {
    cidr_blocks      = []
    description      = "tcp-ipv6-egress"
    from_port        = 0
    ipv6_cidr_blocks = [
      "::/0",
    ]
    prefix_list_ids  = []
    protocol         = "tcp"
    security_groups  = []
    self             = false
    to_port          = 65535
  }

  ingress {
      cidr_blocks = [
        "0.0.0.0/0"
      ]
      from_port = 0
      protocol = -1
      to_port = 0
    }
  ingress {
    cidr_blocks = [
      "0.0.0.0/0"
    ]
    description      = ""
    from_port        = 0
    ipv6_cidr_blocks = []
    prefix_list_ids  = []
    protocol         = "tcp"
    security_groups  = []
    self             = false
    to_port          = 65535
  }
  ingress {
    cidr_blocks      = []
    description      = ""
    from_port        = 0
    ipv6_cidr_blocks = [
      "::/0",
    ]
    prefix_list_ids  = []
    protocol         = "-1"
    security_groups  = []
    self             = false
    to_port          = 0
  }
  ingress {
    cidr_blocks      = []
    description      = ""
    from_port        = 0
    ipv6_cidr_blocks = [
      "::/0",
    ]
    prefix_list_ids  = []
    protocol         = "tcp"
    security_groups  = []
    self             = false
    to_port          = 65535
  }
  vpc_id = data.terraform_remote_state.vpc.outputs.main_vpc_id
  tags = {
    Name = "mainVpc/ChembertaNodeSg"
  }
}

resource "aws_instance" "node1" {
  ami = var.ubuntu_2204_ami 
  # instance_type = "c5.xlarge"
  instance_type = "g4dn.xlarge"
  disable_api_termination = "false"
  associate_public_ip_address = "true"

  subnet_id = data.terraform_remote_state.vpc.outputs.dmz_subnet1_id
  vpc_security_group_ids = [aws_security_group.chemberta_node_sg.id]
  key_name = var.dev_key_pair

  root_block_device {
    encrypted = "true"
    volume_size = 250
    volume_type = "gp3"
  }

  tags = {
    Name = "ChembertaTrainingNode1"
    Project = "Chemberta3"
  }

  iam_instance_profile = "${aws_iam_instance_profile.node_profile.name}"
  ipv6_address_count = 1

  lifecycle {
    ignore_changes = [
      associate_public_ip_address
    ]
  }
}

resource "aws_instance" "node2" {
  ami = var.ubuntu_2204_ami
  # instance_type = "c5.xlarge"
  instance_type = "g4dn.xlarge"
  disable_api_termination = "false"
  associate_public_ip_address = "true"

  subnet_id = data.terraform_remote_state.vpc.outputs.dmz_subnet1_id
  vpc_security_group_ids = [aws_security_group.chemberta_node_sg.id]
  key_name = var.dev_key_pair

  root_block_device {
    encrypted = "true"
    volume_size = 250
    volume_type = "gp3"
  }

  tags = {
    Name = "ChembertaTrainingNode2"
    Project = "Chemberta3"
  }

  iam_instance_profile = "${aws_iam_instance_profile.node_profile.name}"
  ipv6_address_count = 1

  lifecycle {
    ignore_changes = [
      associate_public_ip_address
    ]
  }
}

resource "aws_iam_instance_profile" "node_profile" {
  name_prefix = "NodeProfile"
  role = aws_iam_role.node_role.name
}
