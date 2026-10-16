data "archive_file" "ingest" {
  type        = "zip"
  source_dir  = "${path.module}/../../src"
  output_path = "${path.module}/build/ingest.zip"
  excludes    = ["ingest/__pycache__"]
}

data "aws_ssm_parameter" "odds_api_key" {
  name            = var.odds_api_key_param
  with_decryption = false # only need the ARN; the Lambda decrypts at runtime
}

# ---------- IAM: least privilege for the ingest functions ----------

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingest" {
  name               = "${var.project}-ingest-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "ingest_logs" {
  role       = aws_iam_role.ingest.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "ingest" {
  statement {
    sid       = "WriteRawZone"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.lake.arn}/raw/*"]
  }
  statement {
    # Needed so HeadObject on a missing key returns 404 (not 403).
    sid       = "ListRawZone"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.lake.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["raw/*"]
    }
  }
  statement {
    sid       = "ReadOddsApiKey"
    actions   = ["ssm:GetParameter"]
    resources = [data.aws_ssm_parameter.odds_api_key.arn]
  }
}

resource "aws_iam_role_policy" "ingest" {
  name   = "ingest"
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

# ---------- Functions ----------

locals {
  functions = {
    odds = {
      handler = "ingest.odds.handler"
      env = {
        ODDS_API_KEY_PARAM    = var.odds_api_key_param
        MARKETS               = "player_points"
        REGIONS               = "us"
        LOOKAHEAD_HOURS       = "24"
        MIN_CREDITS_REMAINING = "25"
      }
    }
    boxscores = {
      handler = "ingest.boxscores.handler"
      env     = { LOOKBACK_DAYS = "3" }
    }
  }
}

resource "aws_cloudwatch_log_group" "ingest" {
  for_each          = local.functions
  name              = "/aws/lambda/${var.project}-${each.key}"
  retention_in_days = 30
}

resource "aws_lambda_function" "ingest" {
  for_each = local.functions

  function_name    = "${var.project}-${each.key}"
  role             = aws_iam_role.ingest.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = each.value.handler
  filename         = data.archive_file.ingest.output_path
  source_code_hash = data.archive_file.ingest.output_base64sha256
  timeout          = 300
  memory_size      = 256

  environment {
    variables = merge(each.value.env, { OUTPUT_URI = "s3://${aws_s3_bucket.lake.bucket}" })
  }

  depends_on = [aws_cloudwatch_log_group.ingest, aws_iam_role_policy_attachment.ingest_logs]
}
