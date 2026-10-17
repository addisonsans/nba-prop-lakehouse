# ---------- Push new lake files into Databricks ----------
#
# The project's managed resource control policy blocks Databricks from reading the
# bucket directly, so S3 ObjectCreated events trigger a Lambda that copies each file
# into a Unity Catalog volume via the Databricks Files API.

data "aws_ssm_parameter" "databricks_sync_secret" {
  name            = var.databricks_secret_param
  with_decryption = false # only need the ARN; the Lambda decrypts at runtime
}

resource "aws_iam_role" "databricks_sync" {
  name               = "${var.project}-databricks-sync-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "databricks_sync_logs" {
  role       = aws_iam_role.databricks_sync.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "databricks_sync" {
  statement {
    sid       = "ReadLake"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.lake.arn}/*"]
  }
  statement {
    sid       = "ReadDatabricksSecret"
    actions   = ["ssm:GetParameter"]
    resources = [data.aws_ssm_parameter.databricks_sync_secret.arn]
  }
}

resource "aws_iam_role_policy" "databricks_sync" {
  name   = "databricks-sync"
  role   = aws_iam_role.databricks_sync.id
  policy = data.aws_iam_policy_document.databricks_sync.json
}

resource "aws_cloudwatch_log_group" "databricks_sync" {
  name              = "/aws/lambda/${var.project}-databricks-sync"
  retention_in_days = 30
}

resource "aws_lambda_function" "databricks_sync" {
  function_name    = "${var.project}-databricks-sync"
  role             = aws_iam_role.databricks_sync.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "ingest.databricks_sync.handler"
  filename         = data.archive_file.ingest.output_path
  source_code_hash = data.archive_file.ingest.output_base64sha256
  timeout          = 120
  memory_size      = 256

  environment {
    variables = {
      DATABRICKS_HOST         = var.databricks_host
      DATABRICKS_CLIENT_ID    = var.databricks_client_id
      DATABRICKS_SECRET_PARAM = var.databricks_secret_param
      VOLUME_PATH             = var.databricks_volume_path
    }
  }

  depends_on = [aws_cloudwatch_log_group.databricks_sync, aws_iam_role_policy_attachment.databricks_sync_logs]
}

resource "aws_lambda_permission" "s3_invoke_databricks_sync" {
  statement_id   = "AllowS3Invoke"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.databricks_sync.function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.lake.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "lake" {
  bucket = aws_s3_bucket.lake.id

  dynamic "lambda_function" {
    for_each = ["raw/", "historical/"]
    content {
      lambda_function_arn = aws_lambda_function.databricks_sync.arn
      events              = ["s3:ObjectCreated:*"]
      filter_prefix       = lambda_function.value
    }
  }

  depends_on = [aws_lambda_permission.s3_invoke_databricks_sync]
}
