"""Push lake files from S3 into a Databricks Unity Catalog volume.

Why push: this AWS project's managed resource control policy denies S3 and STS
calls from principals outside the project's organization, so Databricks can't
read the bucket (no storage credential / external location). Code running inside
the project can still call out, so S3 ObjectCreated events trigger this function,
which copies each new object to the same path under the volume via the Databricks
Files API, authenticating as a service principal (OAuth machine-to-machine).

Env:
  DATABRICKS_HOST          https://dbc-....cloud.databricks.com
  DATABRICKS_CLIENT_ID     service principal application id
  DATABRICKS_SECRET_PARAM  SSM SecureString holding its OAuth secret
  DATABRICKS_SECRET        (local runs only) the secret itself, overrides SSM
  VOLUME_PATH              e.g. /Volumes/nba/raw/landing

One-time copy of everything already in the bucket (local run):
  python -m ingest.databricks_sync --bucket <bucket> --prefix raw/ --prefix historical/
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .http import USER_AGENT, HTTPStatusError

log = logging.getLogger()
log.setLevel(logging.INFO)

_token: dict = {}


def _call(method: str, url: str, *, data: bytes | None = None, headers: dict | None = None,
          retries: int = 4, backoff: float = 1.0) -> bytes:
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    host = urllib.parse.urlsplit(url).netloc
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if not (exc.code == 429 or exc.code >= 500) or attempt == retries - 1:
                raise HTTPStatusError(f"HTTP {exc.code} from {host}: {exc.read()[:200]!r}", exc.code) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Network error calling {host}: {exc}") from None
        time.sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")


def _secret() -> str:
    if os.environ.get("DATABRICKS_SECRET"):
        return os.environ["DATABRICKS_SECRET"]
    import boto3

    resp = boto3.client("ssm").get_parameter(Name=os.environ["DATABRICKS_SECRET_PARAM"], WithDecryption=True)
    return resp["Parameter"]["Value"]


def access_token(host: str, client_id: str) -> str:
    """OAuth client-credentials token, cached until a minute before expiry."""
    if _token.get("value") and _token["expires_at"] > time.time() + 60:
        return _token["value"]
    basic = base64.b64encode(f"{client_id}:{_secret()}".encode()).decode()
    body = _call("POST", f"{host}/oidc/v1/token",
                 data=urllib.parse.urlencode({"grant_type": "client_credentials", "scope": "all-apis"}).encode(),
                 headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})
    payload = json.loads(body)
    _token.update(value=payload["access_token"], expires_at=time.time() + int(payload.get("expires_in", 3600)))
    return _token["value"]


def upload(host: str, token: str, volume_path: str, key: str, body: bytes) -> str:
    """PUT a file into the volume at volume_path/key (parent directories are created)."""
    path = f"{volume_path.rstrip('/')}/{key}"
    _call("PUT", f"{host}/api/2.0/fs/files{urllib.parse.quote(path)}?overwrite=true", data=body,
          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"})
    return path


def _config() -> tuple[str, str, str]:
    return os.environ["DATABRICKS_HOST"].rstrip("/"), os.environ["DATABRICKS_CLIENT_ID"], os.environ["VOLUME_PATH"]


def handler(event, context):
    import boto3

    s3 = boto3.client("s3")
    host, client_id, volume = _config()
    copied = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        copied.append(upload(host, access_token(host, client_id), volume, key, body))
    log.info("copied %s", copied)
    return {"copied": len(copied)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Copy existing S3 lake files into the Databricks volume.")
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", action="append", default=[], help="repeatable; default raw/ and historical/")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import boto3

    s3 = boto3.client("s3")
    host, client_id, volume = _config()
    keys = [o["Key"] for prefix in (args.prefix or ["raw/", "historical/"])
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=prefix)
            for o in page.get("Contents", [])]
    log.info("copying %d files", len(keys))

    def copy(key: str) -> str:
        body = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
        return upload(host, access_token(host, client_id), volume, key, body)

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(copy, keys), 1):
            if i % 500 == 0:
                log.info("%d/%d", i, len(keys))
    log.info("done: %d files", len(keys))


if __name__ == "__main__":
    main()
