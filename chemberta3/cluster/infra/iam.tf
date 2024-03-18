resource "aws_iam_role" "dc_multi_node_training_instance_role" {
  name_prefix = "dc_multi_node_training_instance_role"
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
  description = "instance role for giving power permissions for multi-node training in DeepChem"
}

resource "aws_iam_role_policy_attachment" "dc_multi_node_training_instance_s3_access" {
  role = aws_iam_role.dc_multi_node_training_instance_role.id
  for_each = toset(["arn:aws:iam::aws:policy/AmazonS3FullAccess",
										"arn:aws:iam::aws:policy/AmazonEC2FullAccess",
                    ])
  policy_arn = each.value
}
