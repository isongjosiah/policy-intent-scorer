**\# Policy Intent Score System**

**This project implements a \*\*real-time Policy Intent Score system\*\* designed to evaluate White House press releases and predict whether they will lead to a tangible outcome ("Actionable") or are merely a "Bluff". The system is developed as a Minimum Viable Product (MVP) with a target accuracy of over 70%.**

**The project leverages a modular pipeline for data ingestion, heuristic-based labeling, machine learning model training, and real-time prediction serving via a FastAPI API. The underlying infrastructure is automated using Terraform for consistent and reproducible deployments on AWS.**

**\-----**

**\## 1\\. Project Architecture**

**The system's pipeline is engineered for cost-efficiency and scalability, primarily utilizing serverless and managed AWS services.**

**\* \*\*Data Ingestion (AWS Lambda)\*\*: The \`ingest_lambda.py\` script, deployed as an AWS Lambda function, scrapes press releases from the official White House website (\`https://www.whitehouse.gov/briefing-room/statements-and-releases/\`) and falls back to its RSS feed (\`https://www.whitehouse.gov/briefing-room/feed/\`) if HTML parsing fails. It performs a 1-year historical backfill and stores the raw data as \*\*partitioned Parquet files\*\* (by \`year/month/day\`) in an S3 raw data bucket.**

**\* \*\*Heuristic Labeling (AWS Lambda)\*\*: The \`labeler.py\` script, also an AWS Lambda function, is triggered by an S3 event notification whenever new data is saved to the raw S3 bucket. It applies "Actionable" or "Bluff" labels using two heuristics:**

**\* \*\*Congress.gov API\*\*: Checks if a related bill became law within 180 days of the press release.**

**\* \*\*Market Volatility\*\*: Monitors significant market volatility spikes (greater than 2 standard deviations) within 24 hours post-announcement using the \`yfinance\` library for specified sectors (SPY, XLI, XLE, XLF).**

**The labeled data is then saved to a separate S3 processed data bucket.**

**\* \*\*Model Training (Local Execution)\*\*: The \`train_model.py\` script is designed to be run \*\*locally\*\*. It loads the labeled data from the S3 processed bucket, preprocesses the text using \*\*TF-IDF (Term Frequency-Inverse Document Frequency)\*\*, and trains a \*\*Logistic Regression classifier\*\*. The trained model is then serialized using \`pickle\` and uploaded as \`model.pkl\` to an S3 model bucket. The target validation ROC-AUC score is \\> 0.7.**

**\* \*\*Real-time Scoring API (FastAPI on ECS EC2)\*\*: The \`score_api.py\` is a FastAPI application that serves real-time predictions. It's containerized and deployed on an \*\*Amazon ECS cluster backed by EC2 instances\*\* (leveraging the AWS Free Tier). An \*\*API Gateway\*\* acts as the public front door for the API, handling requests and routing them to the ECS service. Rate limiting is implemented using \`fastapi-limiter\` backed by a Redis instance.**

**\-----**

**\## 2\\. Prerequisites**

**Before you begin, ensure you have the following software installed and configured:**

**\### Software Requirements**

**\* \*\*Python\*\*: Version \`3.9\` or higher, but less than \`4.0\` (\`>=3.9, <4.0\`).**

**\* \*\*Poetry\*\*: A Python dependency management and packaging tool.**

**\* Installation: \`curl -sSL https://install.python-poetry.org | python3 -\`**

**\* \*\*Docker\*\*: Used to run a local Redis instance for API rate limiting.**

**\* Installation: Follow instructions on the \[Docker website\](https://docs.docker.com/get-docker/).**

**\* \*\*Terraform\*\*: An Infrastructure as Code (IaC) tool for provisioning AWS resources.**

**\* Installation: Follow instructions on the \[Terraform website\](https://developer.hashicorp.com/terraform/downloads).**

**\* \*\*AWS CLI\*\*: The command-line interface for interacting with AWS services.**

**\* Installation: Follow instructions on the \[AWS CLI documentation\](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).**

**\### AWS Account Setup**

**1\. \*\*Create an AWS Account\*\*: If you don't have one, sign up at \[aws.amazon.com\](https://aws.amazon.com/). New accounts often qualify for a 12-month free tier, which is beneficial for this project's EC2 usage.**

**2\. \*\*Configure AWS CLI\*\*: After creating your account, configure the AWS CLI with your credentials.**

**\`\`\`bash**

**aws configure**

**\`\`\`**

**You will be prompted to enter your AWS Access Key ID, Secret Access Key, default region (e.g., \`us-east-1\`), and default output format.**

**\-----**

**\## 3\\. Local Setup**

**Follow these steps to set up the project on your local machine:**

**1\. \*\*Clone the Repository\*\*:**

**\`\`\`bash**

**git clone https://github.com/your\_username/policy-intent-score.git**

**cd policy-intent-score**

**\`\`\`**

**2\. \*\*Configure Poetry for In-Project Virtual Environment\*\*:**

**It's recommended to have Poetry create the virtual environment directly within your project directory.**

**\`\`\`bash**

**poetry config virtualenvs.in-project true**

**\`\`\`**

**3\. \*\*Install Project Dependencies\*\*:**

**This command will read \`pyproject.toml\` and install all required Python packages into a new virtual environment (\`.venv/\`).**

**\`\`\`bash**

**poetry install**

**\`\`\`**

**4\. \*\*Create Project Directories and Placeholder Files\*\*:**

**Ensure the necessary directories and empty Python files exist for packaging Lambda functions.**

**\`\`\`bash**

**mkdir -p scripts api model data**

**touch scripts/ingest_lambda.py scripts/labeler.py scripts/train_model.py api/score_api.py**

**\`\`\`**

**\-----**

**\## 4\\. Running the Project**

**This section details the steps to deploy and run the entire Policy Intent Score system.**

**\### 4.1. Infrastructure Deployment (Terraform)**

**Navigate to the \`terraform/\` directory and deploy your AWS infrastructure. This will provision S3 buckets, IAM roles, Lambda functions, a VPC, subnets, security groups, an ECS cluster with an Auto Scaling Group for EC2 instances, and an API Gateway.**

**1\. \*\*Navigate to Terraform Directory\*\*:**

**\`\`\`bash**

**cd terraform**

**\`\`\`**

**2\. \*\*Initialize Terraform\*\*:**

**\`\`\`bash**

**terraform init**

**\`\`\`**

**3\. \*\*Review Proposed Changes (Optional but Recommended)\*\*:**

**\`\`\`bash**

**terraform plan**

**\`\`\`**

**4\. \*\*Apply Terraform Configuration\*\*:**

**This will create all the AWS resources. The \`-auto-approve\` flag bypasses the manual confirmation prompt.**

**\`\`\`bash**

**terraform apply -auto-approve**

**\`\`\`**

**Keep note of the S3 bucket names and API Gateway URL from the \`terraform apply\` output.**

**\### 4.2. Prepare Lambda Deployment Packages**

**Return to the project root directory. The Lambda functions require \`.zip\` deployment packages. The \`Makefile\` automates this.**

**1\. \*\*Navigate to Project Root\*\*:**

**\`\`\`bash**

**cd .. # if you are in the terraform directory**

**\`\`\`**

**2\. \*\*Package Lambda Functions\*\*:**

**\`\`\`bash**

**make package-lambdas**

**\`\`\`**

**This command will create \`ingest_lambda.zip\` and \`labeler.zip\` in your project root.**

**3\. \*\*Manually Upload Lambda Zips (Temporary Step)\*\*:**

**Currently, Terraform creates the Lambda function resources, but you need to manually upload the \`.zip\` files to them.**

**\* Go to the AWS Lambda console.**

**\* Find your \`policy-intent-score-ingest-lambda\` function.**

**\* Under the "Code" tab, click "Upload from" -\\> ".zip file" and upload \`ingest_lambda.zip\`.**

**\* Repeat for \`policy-intent-score-labeler-lambda\` with \`labeler.zip\`.**

**\* \*\*Future Improvement\*\*: This step can be automated in Terraform using \`aws_lambda_function.filename\` and \`aws_s3_bucket_object\` resources for S3-backed Lambda deployments.**

**\### 4.3. Run Local Redis for API Rate Limiting**

**The FastAPI application uses Redis for rate limiting. For local development, run a Redis container.**

**1\. \*\*Start Redis Container\*\*:**

**\`\`\`bash**

**docker run --name my-redis -p 6379:6379 -d redis**

**\`\`\`**

**This makes Redis accessible at \`redis://localhost:6379\`.**

**\### 4.4. Ingest and Label Data**

**Now, run the Python scripts to populate your S3 buckets with raw and labeled data.**

**1\. \*\*Set Environment Variables\*\*:**

**You need to set environment variables for your S3 buckets and Congress.gov API key. Replace \`your-raw-data-bucket\`, \`your-processed-data-bucket\`, and \`your-congress-api-key\` with your actual bucket names and API key.**

**\`\`\`bash**

**export S3_RAW_BUCKET="your-raw-data-bucket-name"**

**export S3_PROCESSED_BUCKET="your-processed-data-bucket-name"**

**export CONGRESS_API_KEY="your-congress-gov-api-key"**

**\`\`\`**

**2\. \*\*Run Ingestion Lambda (Locally for testing)\*\*:**

**This will scrape data and save it to your raw S3 bucket.**

**\`\`\`bash**

**poetry run python scripts/ingest_lambda.py**

**\`\`\`**

**\*Note\*: In a production AWS environment, this Lambda would be triggered by a schedule (e.g., EventBridge).**

**3\. \*\*Run Labeler Lambda (Locally for testing)\*\*:**

**This will read from the raw S3 bucket, apply labels, and save to the processed S3 bucket.**

**\`\`\`bash**

**poetry run python scripts/labeler.py**

**\`\`\`**

**\*Note\*: In a production AWS environment, this Lambda would be triggered by an S3 event notification from the raw bucket.**

**\### 4.5. Train and Save the Model**

**Once you have a sufficient amount of labeled data in your processed S3 bucket (the brief suggests 2,000+ samples), you can train the machine learning model.**

**1\. \*\*Set Model S3 Bucket Environment Variable\*\*:**

**\`\`\`bash**

**export S3_MODEL_BUCKET="your-model-bucket-name"**

**\`\`\`**

**2\. \*\*Run Model Training Script\*\*:**

**\`\`\`bash**

**poetry run python scripts/train_model.py**

**\`\`\`**

**This script will load data from \`S3_PROCESSED_BUCKET\`, train the model, and save \`model.pkl\` to \`S3_MODEL_BUCKET\`.**

**\### 4.6. Run the FastAPI Server Locally**

**Finally, start the FastAPI application.**

**1\. \*\*Ensure \`model.pkl\` is Accessible Locally\*\*:**

**For local testing, the \`score_api.py\` expects \`model/model.pkl\` to be present in the local \`model/\` directory. You'll need to manually download the \`model.pkl\` file from your S3 model bucket to \`policy-intent-score/model/\` for local testing.**

**2\. \*\*Run the FastAPI Application\*\*:**

**\`\`\`bash**

**poetry run uvicorn api.score_api:app --reload --host 0.0.0.0 --port 8000**

**\`\`\`**

**The API will be accessible at \`http://localhost:8000\`.**

**3\. \*\*Test the API\*\*:**

**You can send a POST request to \`http://localhost:8000/score\` with a JSON body:**

**\`\`\`json**

**{**

**"headline": "President signs new climate bill",**

**"body": "The President today signed a groundbreaking bill aimed at reducing carbon emissions..."**

**}**

**\`\`\`**

**\-----**

**\## 5\\. Next Steps & Future Improvements**

**\* \*\*Automate Lambda Deployment\*\*: Integrate the \`.zip\` file uploads directly into Terraform using S3-backed Lambda deployments.**

**\* \*\*ECS Container Image\*\*: Build a Docker image for the FastAPI application and push it to Amazon ECR, then update the ECS task definition in Terraform to use this image.**

**\* \*\*ElastiCache for Redis\*\*: Provision a managed Redis instance (e.g., Amazon ElastiCache) using Terraform for production rate-limiting.**

**\* \*\*CI/CD Pipeline\*\*: Implement a CI/CD pipeline (e.g., AWS CodePipeline/CodeBuild) to automate testing, packaging, and deployment.**

**\* \*\*Monitoring & Logging\*\*: Set up AWS CloudWatch for monitoring Lambda and ECS, and configure centralized logging.**

**\* \*\*Advanced ML\*\*: Replace TF-IDF with fine-tuned DistilBERT for semantic understanding, as suggested in the brief.**

**\* \*\*Active Learning\*\*: Implement an active learning pipeline to scale labeled samples beyond 10k.**

**\* \*\*Multi-class Labeling\*\*: Extend the labeling to "High/Medium/Low impact" instead of binary.**
