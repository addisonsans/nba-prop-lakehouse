output "bucket" {
  value = aws_s3_bucket.lake.bucket
}

output "functions" {
  value = { for k, f in aws_lambda_function.ingest : k => f.function_name }
}
