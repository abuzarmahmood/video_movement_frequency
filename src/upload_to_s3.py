"""
Upload recent data files to AWS S3 bucket.
Requires AWS credentials to be set in environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
"""

import os
import boto3
from botocore.exceptions import ClientError
import glob
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_aws_credentials():
    """Check if required AWS credentials are set in environment variables."""
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required AWS credentials: {', '.join(missing_vars)}"
        )

def upload_file(file_path, bucket, object_name=None):
    """Upload a file to S3 bucket.
    
    Args:
        file_path (str): Path to file to upload
        bucket (str): Bucket name
        object_name (str): S3 object name. If not specified, file_path is used
    
    Returns:
        bool: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = file_path

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True

def main():
    """Main function to upload recent data files to S3."""
    try:
        # Validate AWS credentials
        validate_aws_credentials()
        
        # Get S3 bucket name from environment
        bucket_name = os.getenv('AWS_S3_BUCKET')
        if not bucket_name:
            raise EnvironmentError("AWS_S3_BUCKET environment variable not set")

        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        recent_data_dir = os.path.join(base_dir, 'artifacts', 'recent_data')
        
        # Get current timestamp for folder organization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Upload all CSV files in recent_data directory
        csv_files = glob.glob(os.path.join(recent_data_dir, '*.csv'))
        
        for file_path in csv_files:
            # Create S3 object name with timestamp folder
            file_name = os.path.basename(file_path)
            object_name = f"recent_data/{timestamp}/{file_name}"
            
            # Upload file
            if upload_file(file_path, bucket_name, object_name):
                logger.info(f"Successfully uploaded {file_name} to {bucket_name}/{object_name}")
            else:
                logger.error(f"Failed to upload {file_name}")

    except Exception as e:
        logger.error(f"Error during upload process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
