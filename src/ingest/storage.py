"""Write gzipped raw JSON to S3 or a local folder.

OUTPUT_URI decides the backend:
  s3://bucket[/prefix]  -> S3 (what the Lambdas use)
  /some/local/path      -> local disk (tests and local backfills)
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def envelope(data: object, *, source: str, params: dict | None = None, extra: dict | None = None) -> dict:
    """Wrap a raw API response with provenance metadata (never includes API keys)."""
    safe_params = {k: v for k, v in (params or {}).items() if k.lower() not in {"apikey", "api_key"}}
    meta = {"source": source, "params": safe_params, "fetched_at": utc_now_iso(), "ingest_version": 1}
    meta.update(extra or {})
    return {"_meta": meta, "data": data}


class Store:
    def __init__(self, uri: str):
        self.uri = uri
        if uri.startswith("s3://"):
            import boto3  # available in the Lambda runtime; only needed for S3

            self._s3 = boto3.client("s3")
            self.bucket, _, prefix = uri[len("s3://"):].partition("/")
            self.prefix = prefix.strip("/")
        else:
            self._s3 = None
            self.root = Path(uri)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        if self._s3 is None:
            return (self.root / key).exists()
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def get_json(self, key: str) -> object:
        if self._s3 is None:
            body = (self.root / key).read_bytes()
        else:
            body = self._s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        return json.loads(gzip.decompress(body))

    def put_json(self, key: str, obj: object) -> str:
        body = gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"))
        if self._s3 is None:
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            return str(path)
        self._s3.put_object(
            Bucket=self.bucket, Key=self._key(key), Body=body, ContentType="application/gzip"
        )
        return f"s3://{self.bucket}/{self._key(key)}"
