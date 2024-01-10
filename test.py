import json
import boto3
import base64

# import requests
s3_client = boto3.client('s3')

def lambda_handler(event, context):
  
        file_path = "train/Airplane/attack_aircraft_s_001210.png"
        bucket = "sdl-cifar10"
        data = s3_client.get_object(Bucket=bucket, Key=file_path)
        image_content_bytes = data['Body'].read()

        # Encode image data in base64
        image_content_base64 = base64.b64encode(image_content_bytes).decode('utf-8')


        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "file retrieved",
                "contents": image_content_base64
                # "location": ip.text.replace("\n", "")
            }),
        }

if __name__ == "__main__":
   response = lambda_handler(None, None)
   print(response)

