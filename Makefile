.PHONY: all setup clean package-lambdas deploy

# The default target. Runs a full setup and deployment.
all: setup package-lambdas

# Sets up the project by installing dependencies.
setup:
	@echo "Setting up project dependencies with Poetry..."
	poetry install

# Cleans up the build artifacts.
clean:
	@echo "Cleaning up build artifacts..."
	rm -f ingest_lambda.zip
	rm -f labeler.zip

# Packages the Lambda functions into zip files.
# This target is crucial for the Terraform deployment.
package-lambdas: clean
	@echo "Packaging Lambda functions..."
	zip -j terraform/lambda/ingest_lambda.zip scripts/ingest_lambda.py
	zip -j terraform/lambda/labeler.zip scripts/labeler.py

# A placeholder for the Terraform deployment command.
# This will apply the infrastructure defined in the terraform/ directory.
deploy:
	@echo "Deploying infrastructure with Terraform..."
	cd terraform && terraform init
	cd terraform && terraform apply -auto-approve
