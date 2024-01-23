#!/bin/bash

sudo apt update
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

sudo apt -y install python3-pip
sudo -u ubuntu bash -c "pip3 install boto3 botocore"
aws s3 cp s3://chemberta3/make_ten.py /home/ubuntu/make_ten.py
sudo -u ubuntu bash -c "python3 /home/ubuntu/make_ten.py"

# TODO Update this to query region from metadata
AWS_REGION=us-east-2
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
FLEET_ID=$(aws ec2 describe-instances --instance-id $INSTANCE_ID --query "Reservations[*].Instances[*].Tags[?Key=='aws:ec2:fleet-id'].Value[]" --output text)
aws ec2 delete-fleets ---region $AWS_REGION --fleet-id $FLEET_ID --terminate-instances
