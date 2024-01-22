aws ec2 describe-instances --filters "Name=tag:Name,Values=SpotJobInstances" --query "Reservations[*].Instances[*].{State:State,id:InstanceId}"
# aws ec2 describe-instances --filters "Name=tag:Name,Values=TestJob" --query "Reservations[*].Instances[*][][].PublicIpAddress"
