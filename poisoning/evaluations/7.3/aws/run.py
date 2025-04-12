from cohere_aws import Client
import boto3

# Explicitly set the region
region = "us-east-1"

cohere_package = "cohere-embed-english-v3-8-0117-6d3f676bd44232428f648851d90f9f9c"

# Mapping for Model Packages
model_package_map = {
    "us-east-1": f"arn:aws:sagemaker:us-east-1:865070037744:model-package/{cohere_package}",
    "us-east-2": f"arn:aws:sagemaker:us-east-2:057799348421:model-package/{cohere_package}",
    "us-west-1": f"arn:aws:sagemaker:us-west-1:382657785993:model-package/{cohere_package}",
    "us-west-2": f"arn:aws:sagemaker:us-west-2:594846645681:model-package/{cohere_package}",
    "ca-central-1": f"arn:aws:sagemaker:ca-central-1:470592106596:model-package/{cohere_package}",
    "eu-central-1": f"arn:aws:sagemaker:eu-central-1:446921602837:model-package/{cohere_package}",
    "eu-west-1": f"arn:aws:sagemaker:eu-west-1:985815980388:model-package/{cohere_package}",
    "eu-west-2": f"arn:aws:sagemaker:eu-west-2:856760150666:model-package/{cohere_package}",
    "eu-west-3": f"arn:aws:sagemaker:eu-west-3:843114510376:model-package/{cohere_package}",
    "eu-north-1": f"arn:aws:sagemaker:eu-north-1:136758871317:model-package/{cohere_package}",
    "ap-southeast-1": f"arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/{cohere_package}",
    "ap-southeast-2": f"arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/{cohere_package}",
    "ap-northeast-2": f"arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/{cohere_package}",
    "ap-northeast-1": f"arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/{cohere_package}",
    "ap-south-1": f"arn:aws:sagemaker:ap-south-1:077584701553:model-package/{cohere_package}",
    "sa-east-1": f"arn:aws:sagemaker:sa-east-1:270155090741:model-package/{cohere_package}",
}

# Check if region is supported
if region not in model_package_map.keys():
    raise Exception(f"Current boto3 session region {region} is not supported.")

model_package_arn = model_package_map[region]

# Initialize boto3 session with explicit region
boto3_session = boto3.Session(region_name=region)

# Proceed with your endpoint creation or connection
co = Client(region_name=region)
# Example: Connect to existing endpoint
co.connect_to_endpoint(endpoint_name="cohere-embed-english-v3")