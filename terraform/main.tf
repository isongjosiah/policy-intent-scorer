terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Buckets for data storage
resource "aws_s3_bucket" "raw_data_bucket" {
  bucket = "${var.project_name}-raw-data"
}

resource "aws_s3_bucket" "processed_data_bucket" {
  bucket = "${var.project_name}-processed-data"
}

# IAM Roles for Lambda and EC2
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com",
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_policy" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ec2_instance_profile" {
  name = "${var.project_name}-ec2-instance-profile"
  role = aws_iam_role.ec2_role.name
}

# Lambda Functions for ingestion and labeling
resource "aws_lambda_function" "ingest_lambda" {
  function_name    = "${var.project_name}-ingest-lambda"
  role             = aws_iam_role.lambda_role.arn
  handler          = "ingest_lambda.handler"
  runtime          = "python3.9"
  filename         = "ingest_lambda.zip"
  source_code_hash = filebase64sha256("./lambda/ingest_lambda.zip")
}

resource "aws_lambda_function" "labeler_lambda" {
  function_name    = "${var.project_name}-labeler-lambda"
  role             = aws_iam_role.lambda_role.arn
  handler          = "labeler.handler"
  runtime          = "python3.9"
  filename         = "labeler.zip"
  source_code_hash = filebase64sha256("./lambda/labeler.zip")
}

# Networking and ECS for the API
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "main" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = "${var.aws_region}${count.index == 0 ? "a" : "b"}"
}

resource "aws_security_group" "ecs_security_group" {
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
}

resource "aws_launch_template" "ecs_launch_template" {
  name_prefix   = "${var.project_name}-launch-template-"
  image_id      = "ami-0c55b159cbfafe1f0" # ECS-optimized AMI
  instance_type = "t2.micro"              # Free tier eligible
  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_instance_profile.name
  }
}

resource "aws_autoscaling_group" "ecs_asg" {
  name                = "${var.project_name}-asg"
  vpc_zone_identifier = aws_subnet.main[*].id
  desired_capacity    = 1
  min_size            = 1
  max_size            = 3
  launch_template {
    id      = aws_launch_template.ecs_launch_template.id
    version = "$$Latest"
  }
}

resource "aws_ecs_task_definition" "main" {
  family                   = "${var.project_name}-task"
  requires_compatibilities = ["EC2"]
  cpu                      = "256"
  memory                   = "512"
  network_mode             = "awsvpc"
  execution_role_arn       = aws_iam_role.ecs_task_role.arn
  container_definitions = jsonencode([
    {
      name  = "${var.project_name}-container",
      image = "nginx:latest",
      portMappings = [
        {
          containerPort = 8000,
          hostPort      = 8000
        }
      ]
    }
  ])
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_ecs_service" "main" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = 1
  launch_type     = "EC2"
  network_configuration {
    subnets         = aws_subnet.main[*].id
    security_groups = [aws_security_group.ecs_security_group.id]
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "main" {
  name = "${var.project_name}-api"
}

resource "aws_api_gateway_resource" "score_resource" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "score"
}

resource "aws_api_gateway_method" "score_method" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.score_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

