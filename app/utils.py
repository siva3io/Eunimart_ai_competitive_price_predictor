
import logging
import json
import boto3
from config import Config

def catch_exceptions(func):
    def wrapped_function(*args, **kargs):
        try:
            return func(*args, **kargs)
        except Exception as e:
            l = logging.getLogger(func.__name__)
            l.error(e, exc_info=True)
            return None                
    return wrapped_function



boto3_session = boto3.session.Session(
            aws_access_key_id=Config.BOTO3_ACCESS_KEY,
            aws_secret_access_key=Config.BOTO3_SECRET_KEY)
s3 = boto3_session.resource('s3')

def download_from_s3(source_path,destination_path):
    s3.Bucket(Config.BOTO3_BUCKET).download_file(source_path,destination_path)