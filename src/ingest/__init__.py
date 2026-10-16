"""Raw (bronze) ingestion for the NBA prop lakehouse.

Everything here uses only the standard library plus boto3, which the AWS
Lambda Python runtime already ships — so the Lambda zip has no dependencies.
"""
