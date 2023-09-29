terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  profile = "default"
  region  = "us-east-2"
}

resource "aws_security_group" "chemberta_instance_sg" {
  name = "mainVpc/chembertaInstanceSg"
  egress {
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
      from_port = 0
      protocol = -1
      to_port = 0
    }
  vpc_id = var.vpc_id
  tags = {
    Name = "mainVpc/ChembertaInstanceSg"
  }
}

resource "aws_iam_instance_profile" "chemberta_instance_profile" {
  name_prefix = "ChembertaInstanceProfile"
  role = aws_iam_role.chemberta_instance_role.name
}

resource "aws_instance" "chemberta_instance" {
  ami = var.ubuntu_2204_ami 
  instance_type = "g3.8xlarge"
  # instance_type = "t3.small"
  disable_api_termination = "true"
  associate_public_ip_address = true
  subnet_id = var.subnet_id 
  vpc_security_group_ids = [aws_security_group.chemberta_instance_sg.id]
  key_name = var.chemberta_key_pair

  root_block_device {
    encrypted = "true"
    volume_size = 250
    volume_type = "gp3"
  }

  tags = {
    Name = "ChembertaInstance"
    Project = "Chemberta3"
  }

  iam_instance_profile = "${aws_iam_instance_profile.chemberta_instance_profile.name}"
}
