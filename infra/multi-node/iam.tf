resource "aws_iam_role" "node_role" {
  name_prefix = "node_power_role"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
  description = "instance role for giving giving permissions to training nodes"
}

resource "aws_iam_role_policy_attachment" "chemberta_instance_s3_access" {
  role = aws_iam_role.node_role.id
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
