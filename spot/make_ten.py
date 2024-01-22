import boto3
import botocore
import time


def get_progress():
    s3_bucket = 'chemberta3'
    s3_key = 'write_ten_checkpoint.txt'
    s3_client = boto3.client('s3')
    # check whether object exists in s3 or not
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        i = int(response['Body'].read().decode('utf-8'))
        return i
    except botocore.exceptions.ClientError as e:
        return 0

def write_checkpoint(i):
    s3_bucket = 'chemberta3'
    s3_key = 'write_ten_checkpoint.txt'
    s3_client = boto3.client('s3')
    response = s3_client.put_object(Body=str(i), Bucket=s3_bucket, Key=s3_key)
    assert response['ResponseMetadata']['HTTPStatusCode'] == 200

def make_i(i):
    s3_bucket = 'chemberta3'
    s3_key = 'tens/' + str(i) + '.txt'
    s3_client = boto3.client('s3')
    response = s3_client.put_object(Body=str(i), Bucket=s3_bucket, Key=s3_key)
    assert response['ResponseMetadata']['HTTPStatusCode'] == 200

def clean():
    s3_bucket = 'chemberta3'
    s3_key = 'write_ten_checkpoint.txt'
    s3_client = boto3.client('s3')
    response = s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
    assert response['ResponseMetadata']['HTTPStatusCode'] == 204

    for i in range(0, 10):
        s3_key = 'tens/' + str(i) + '.txt'
        response = s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
        assert response['ResponseMetadata']['HTTPStatusCode'] == 204


if __name__ == '__main__':
    start = get_progress()
    print ('start value is ', start)
    for i in range(start, 10):
        time.sleep(10)
        make_i(i)
        write_checkpoint(i)
        print ('written ', i)

    # clean()
