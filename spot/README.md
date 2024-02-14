# Notes

## System Setup

The key idea here is to decouple compute, storage and code artifacts. Spot instances perform only the computation - training of models. Code (specifying model) and data artifacts are uploaded to S3 bucket (or an git repository) from where it is downloaded for the commencement/resumption of training. User data specifying the S3 path to training script is created and passed to the instance at the launch for performing training.

A EC2 fleet request specifying the type of instance to use and the bid price is issued for launching and maintaining spot fleet. The fleet request automatically handled launching of instance with the specified user configuration and requirements. When the instance is launched, data and training script are downloaded to the EC2 instance for training. The training progresses and the model checkpoints are periodically saved into a persistent-storage system using Ray.

If instance gets terminated (happens when spot price is higher than our bid price), the EC2 Fleet Request issues request for new instance. When a new instance is allocated (happens when spot price goes lower than our bid price), training and code artifacts are once again downloaded from S3 and training resumes from the checkpoint.

Once the training is completed, the fleet request is automatically deleted.

## AWS Resources
- [Spot Instance Advisor](https://aws.amazon.com/ec2/spot/instance-advisor/) for finding frequency of interruption of spot instances.
- [Spot Pricing](https://aws.amazon.com/ec2/spot/pricing/) for getting spot instance pricing.
- [EC2 Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) for finding pricing of on-demand instances.
