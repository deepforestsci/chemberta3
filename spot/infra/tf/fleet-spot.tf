resource "aws_iam_instance_profile" "spot_instance_profile" {
  name_prefix = "DcSpotInstance"
  role = aws_iam_role.dc_spot_instance_role.name
}


resource "aws_launch_template" "spot_instance_lt" {
	image_id = var.ubuntu_2204_cpu 
  key_name = var.dc_spot_key_pair
  name_prefix = "test-spot"

  user_data = file("job_base64.sh")

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 10
      volume_type = "gp3"
      delete_on_termination = true
    }
  }
  iam_instance_profile {
    name = aws_iam_instance_profile.spot_instance_profile.name
  }
	
	network_interfaces {	
		subnet_id = var.subnet_id
		security_groups = [aws_security_group.spot_instance_sg.id]
		associate_public_ip_address = true
	}
	
	tag_specifications {
		resource_type = "instance"
		tags = {
  	  Name = "SpotJobInstances"
  	  Project = "dc-spot"
  	}
	} 
}

resource "aws_ec2_fleet" "write_tens" {

  launch_template_config {
		launch_template_specification {
			launch_template_id = aws_launch_template.spot_instance_lt.id
			version = aws_launch_template.spot_instance_lt.latest_version
		}
		override {
			instance_type = "r4.xlarge"
		  max_price = 0.2
		}
		override {
			instance_type = "r4.large"
			max_price = 0.1
		}
  }
	
  spot_options {
		allocation_strategy = "lowestPrice"
		# For persistent data, we can make instance interruption behavior as stop
		# and then use data from root volume.
		instance_interruption_behavior = "terminate"
		instance_pools_to_use_count = 1
  }

  target_capacity_specification {
    default_target_capacity_type = "spot"
		on_demand_target_capacity = 0
    total_target_capacity = 1
  }

  terminate_instances = "true"
  terminate_instances_with_expiration = "true"
  type = "maintain"
}
