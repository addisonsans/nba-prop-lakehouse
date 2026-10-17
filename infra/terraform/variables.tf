variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "nba-lakehouse"
}

variable "region" {
  type    = string
  default = "us-east-2"
}

variable "aws_profile" {
  description = "AWS CLI profile Terraform authenticates with."
  type        = string
  default     = "nba-lakehouse"
}

variable "alert_email" {
  description = "Where budget and pipeline-failure alerts are sent. Set it in terraform.tfvars (gitignored)."
  type        = string
}

variable "monthly_budget_usd" {
  type    = number
  default = 5
}

variable "odds_api_key_param" {
  description = "SSM SecureString parameter holding The Odds API key. Created by hand, never by Terraform, so the key stays out of state."
  type        = string
  default     = "/nba-lakehouse/odds-api-key"
}

variable "odds_schedule" {
  description = "When to snapshot prop lines (America/New_York). Default 5:00pm ET, before the 7pm slate."
  type        = string
  default     = "cron(0 17 * * ? *)"
}

variable "boxscore_schedule" {
  description = "When to ingest finished games (America/New_York). Default 6:00am ET."
  type        = string
  default     = "cron(0 6 * * ? *)"
}

variable "databricks_host" {
  description = "Databricks workspace URL, e.g. https://dbc-xxxx.cloud.databricks.com. Set in terraform.tfvars."
  type        = string
}

variable "databricks_client_id" {
  description = "Application id of the Databricks service principal that writes to the volume. Set in terraform.tfvars."
  type        = string
}

variable "databricks_secret_param" {
  description = "SSM SecureString holding the service principal's OAuth secret. Created by hand, never by Terraform."
  type        = string
  default     = "/nba-lakehouse/databricks-sync-secret"
}

variable "databricks_volume_path" {
  description = "Unity Catalog volume that mirrors the bucket."
  type        = string
  default     = "/Volumes/nba/raw/landing"
}
