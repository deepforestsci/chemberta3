#!/bin/bash

# install aws cli
sudo apt update
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
 
# install dependencies
sudo apt -y install python3-pip
sudo -u ubuntu bash -c "pip3 install boto3 botocore"
sudo -u ubuntu bash -c "pip3 install torch --index-url https://download.pytorch.org/whl/cpu"
sudo -u ubuntu bash -c "pip3 install --pre deepchem"
sudo -u ubuntu bash -c "pip3 install ray[default]"
sudo -u ubuntu bash -c "pip3 install  dgl -f https://data.dgl.ai/wheels/repo.html"
sudo -u ubuntu bash -c "pip3 install dgllife"

# gather data, training script and execute
# copy training script
aws s3 cp s3://chemberta3/spot/train.py /home/ubuntu/train.py
# copy data to instance
aws s3 cp --recursive s3://chemberta3/spot/data /home/ubuntu/data
sudo -u ubuntu bash -c "python3 /home/ubuntu/train.py"

# TODO Update this to query region from metadata
AWS_REGION=us-east-2
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
FLEET_ID=$(aws ec2 describe-instances --instance-id $INSTANCE_ID --query "Reservations[*].Instances[*].Tags[?Key=='aws:ec2:fleet-id'].Value[]" --output text)
aws ec2 delete-fleets ---region $AWS_REGION --fleet-id $FLEET_ID --terminate-instances
