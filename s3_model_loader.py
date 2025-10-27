import boto3
from botocore.exceptions import ClientError
import tempfile
import shutil
import os
import logging

logger = logging.getLogger(__name__)

class S3ModelLoader:
    """Handles loading models from S3 buckets"""
    
    def __init__(self):
        self.s3_client = self._create_s3_client()
        self.local_model_dir = tempfile.mkdtemp(prefix="s3_models_")
        logger.info(f"Created temporary directory for S3 models: {self.local_model_dir}")
    
    def _create_s3_client(self):
        """Create S3 client with environment variables"""
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required for S3 model loading")
        
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        s3_client = session.client(
            's3',
            endpoint_url=aws_endpoint_url
        )
        
        return s3_client
    
    def download_model_from_s3(self, s3_uri: str) -> str:
        """Download model from S3 and return local path"""
        try:
            # Parse S3 URI: s3://bucket-name/path/to/model
            if not s3_uri.startswith("s3://"):
                raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with 's3://'")
            
            # Remove s3:// prefix and split into bucket and key
            path_parts = s3_uri[5:].split('/', 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid S3 URI format: {s3_uri}")
            
            bucket_name, key_prefix = path_parts
            
            # Create local directory for this model
            model_name = key_prefix.replace('/', '_')
            local_model_path = os.path.join(self.local_model_dir, model_name)
            os.makedirs(local_model_path, exist_ok=True)
            
            logger.info(f"Downloading model from S3: {s3_uri} to {local_model_path}")
            
            # List and download all objects with the prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=key_prefix):
                if 'Contents' not in page:
                    raise ValueError(f"No objects found in S3 bucket {bucket_name} with prefix {key_prefix}")
                
                for obj in page['Contents']:
                    object_key = obj['Key']
                    local_file_path = os.path.join(local_model_path, object_key[len(key_prefix):].lstrip('/'))
                    
                    # Create directory structure if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download the file
                    self.s3_client.download_file(bucket_name, object_key, local_file_path)
                    logger.debug(f"Downloaded: {object_key} -> {local_file_path}")
            
            logger.info(f"Successfully downloaded model from S3 to: {local_model_path}")
            return local_model_path
            
        except ClientError as e:
            logger.error(f"S3 download error for {s3_uri}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error downloading model from S3 {s3_uri}: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up temporary model directories"""
        try:
            if os.path.exists(self.local_model_dir):
                shutil.rmtree(self.local_model_dir)
                logger.info(f"Cleaned up temporary S3 model directory: {self.local_model_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary S3 model directory: {str(e)}")