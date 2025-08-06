variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "policy-intent-score"
}

variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}
