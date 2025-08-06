output "raw_data_bucket_name" {
  description = "Name of the raw data S3 bucket"
  value       = aws_s3_bucket.raw_data_bucket.bucket
}

output "processed_data_bucket_name" {
  description = "Name of the processed data S3 bucket"
  value       = aws_s3_bucket.processed_data_bucket.bucket
}

output "api_gateway_url" {
  description = "The URL of the API Gateway"
  value       = aws_api_gateway_rest_api.main.execution_arn
}
