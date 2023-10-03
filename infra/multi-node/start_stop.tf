variable "instance_state" {
  type = string
  # default = "running"
  default = "stopped"
}

resource "aws_ec2_instance_state" "node1" {
  instance_id = aws_instance.node1.id
  state       = var.instance_state 
}

resource "aws_ec2_instance_state" "node2" {
  instance_id = aws_instance.node2.id
  state       = var.instance_state 
}
