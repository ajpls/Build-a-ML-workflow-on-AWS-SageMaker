#serializeImageData

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    filename = "/tmp/image.png"

    #boto3.resource('s3').Bucket(bucket).download_file(key, "./tmp/image.png")
    s3.download_file(bucket, key, filename)

    # We read the data from a file
    #with open("/tmp/image.png", "rb") as f:
    with open(filename, "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


#classifier

import os
import sys
import subprocess

# pip install custom package to /tmp/ and add to path
subprocess.call('pip install sagemaker -t /tmp/ --no-cache-dir'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(1, '/tmp/')

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer


# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2021-11-27-10-54-57-877"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
    
    
#Used link: https://stackoverflow.com/questions/60311148/pip-install-python-package-within-aws-lambda, Accessed 27/11/21

#inference

import json

THRESHOLD = .95 

def lambda_handler(event, context):
    
    meets_threshold = False

    # Grab the inferences from the event
    inferences = json.loads(event["inferences"])

    # Check if any values in our inferences are above THRESHOLD
    for i in inferences:
        if i > THRESHOLD:
            meets_threshold = True
            break
 
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

