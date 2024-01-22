# add option to start, stop, login into the instance

NO_ARGS=0
E_OPTERROR=85

instance_details=$(aws ec2 describe-instances --filters 'Name=tag:Name,Values=FormatVolInstance' --query 'Reservations[*].Instances[*]')
instance_id=$(echo $instance_details | jq -r '.[][].InstanceId')
public_ip=$(echo $instance_details | jq -r '.[][].PublicIpAddress')

if [ $# -eq "$NO_ARGS" ]    # Script invoked with no command-line args?
then
  echo "Usage: `basename $0` options (-sel)"
  exit $E_OPTERROR          # Exit and explain usage.
                            # Usage: scriptname -options
                            # Note: dash (-) necessary
fi  

while getopts "selpti" Option
do
    case $Option in
      s ) aws ec2 start-instances --instance-ids $instance_id;;
      e ) aws ec2 stop-instances --instance-ids $instance_id;;
      l ) ssh -i dev-instance.pem ubuntu@$public_ip;;
      p ) echo $public_ip;;
      t ) aws ec2 describe-instance-status --instance-ids $instance_id;;
      i ) echo "instance id is $instance_id";;
    esac
done
