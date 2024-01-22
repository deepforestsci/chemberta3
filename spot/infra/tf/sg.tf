resource "aws_security_group" "spot_instance_sg" {
  name = "mainVpc/spotInstanceSg"
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
    Name = "mainVpc/spotInstanceSg"
  }
}

