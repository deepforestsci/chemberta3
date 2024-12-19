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

resource "aws_iam_role" "dc_ray_head_node_role" {
  name_prefix = "dc_ray_head_node_role"
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
  description = "EC2 instance role for giving s3 and aws batch access for ray head node"
}

resource "aws_iam_instance_profile" "head_node_profile" {
    name = "dc_ray_head_node_profile"
    role = aws_iam_role.dc_ray_head_node_role.name
}

resource "aws_iam_role" "dc_ray_worker_node_role" {
  name_prefix = "dc_ray_head_worker_role"
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
  description = "EC2 instance role for giving s3 and aws batch access for ray head node"
}

resource "aws_iam_instance_profile" "worker_node_profile" {
    name = "dc_ray_worker_node_profile"
    role = aws_iam_role.dc_ray_worker_node_role.name
}

resource "aws_iam_policy" "dc_ray_s3_access_policy" {
    name_prefix = "dc_ray_s3_access_policy_"
    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "s3:*",
                Effect = "Allow",
                Resource = [
                    "arn:aws:s3:::chemberta3/*",
                    "arn:aws:s3:::chemberta3",
                ],
            },
        ],
    })
}

# Ideally, the policy should only have permissions described in this
# issue: https://github.com/ray-project/ray/issues/9327
resource "aws_iam_policy" "dc_ray_ec2_access_policy" {
    name_prefix = "dc_ray_ec2_access_policy_"
    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "ec2:*",
                Effect = "Allow",
                Resource = "*",
            },
        ],
    })
}

resource "aws_iam_policy" "dc_ray_iam_access_policy" {
    name_prefix = "dc_ray_iam_access_policy_"
    policy = jsonencode({
        Version = "2012-10-17",
        Statement = [
            {
                Action = "iam:*",
                Effect = "Allow",
                Resource = "*",
            },
        ],
    })
}

resource "aws_iam_role_policy_attachment" "dc_ray_head_node_policy_attachment" {
    role = aws_iam_role.dc_ray_head_node_role.name
    for_each = toset([aws_iam_policy.dc_ray_ec2_access_policy.arn,
                      aws_iam_policy.dc_ray_s3_access_policy.arn,
											aws_iam_policy.dc_ray_iam_access_policy.arn,])
    policy_arn = each.value
}

resource "aws_iam_role_policy_attachment" "dc_worker_node_policy_attachment" {
    role = aws_iam_role.dc_ray_worker_node_role.name
    policy_arn = aws_iam_policy.dc_ray_s3_access_policy.arn
}
