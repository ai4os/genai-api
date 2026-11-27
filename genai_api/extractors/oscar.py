import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
import requests
from botocore.exceptions import ClientError as BotoCoreClientError

from genai_api.api.deps import CurrentUser
from genai_api.core.config import settings


@dataclass(frozen=True)
class OSCAR:
    access_token: str
    endpoint: str
    service_name: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_verify: bool
    timeout_seconds: int
    poll_seconds: int

    @classmethod
    def from_env(cls) -> "OSCAR":
        endpoint = settings.oscar_endpoint.rstrip("/")
        service_name = settings.oscar_doc_extractor_service_name
        missing = [
            name
            for name, value in (
                ("access_token", settings.oscar_access_token),
                ("endpoint", endpoint),
                ("service_name", service_name),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing OSCAR configuration: {', '.join(missing)}"
            )
        if (
            settings.oscar_timeout_seconds <= 0
            or settings.oscar_poll_seconds <= 0
        ):
            raise RuntimeError(
                "OSCAR timeout_seconds and poll_seconds must be greater than zero"
            )

        minio_provider = get_oscar_minio_provider(
            endpoint, settings.oscar_access_token
        )
        oscar = cls(
            access_token=settings.oscar_access_token,
            endpoint=endpoint,
            service_name=service_name,
            minio_endpoint=minio_provider["endpoint"],
            minio_access_key=minio_provider["access_key"],
            minio_secret_key=minio_provider["secret_key"],
            minio_verify=minio_provider["verify"],
            timeout_seconds=settings.oscar_timeout_seconds,
            poll_seconds=settings.oscar_poll_seconds,
        )
        return oscar


def oscar_document_extractor(
    content: bytes, filename: str
) -> str:
    """Convert an uploaded document to Markdown using the OSCAR service."""
    oscar = OSCAR.from_env()
    return convert_to_markdown_oscar_job(content, filename, oscar).decode("utf-8")


def get_oscar_minio_provider(
    oscar_endpoint: str, access_token: str
) -> dict[str, str | bool]:
    response = requests.get(
        f"{oscar_endpoint}/system/config",
        timeout=settings.oscar_timeout_seconds,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            "OSCAR returned an unexpected status code "
            f"{response.status_code} while loading MinIO configuration"
        )

    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("[OSCAR] returned an invalid configuration response")

    provider = payload.get("minio_provider")
    if not isinstance(provider, dict):
        raise TypeError("[OSCAR] returned an invalid MinIO configuration")

    required = {
        "endpoint": provider.get("endpoint"),
        "access_key": provider.get("access_key"),
        "secret_key": provider.get("secret_key"),
    }
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise RuntimeError("OSCAR returned incomplete MinIO credentials")

    verify = provider.get("verify", True)
    if not isinstance(verify, bool):
        raise TypeError("OSCAR returned an invalid MinIO verify setting")
    return {**required, "verify": verify}

def convert_to_markdown_oscar_job(
    content: bytes,
    filename: str,
    oscar: OSCAR,
) -> bytes:
    """Upload a document to MinIO, wait for OSCAR, and download the result."""
    input_key, output_key = _doc_esxtraction_keys(filename)
    client = boto3.client(
        "s3",
        endpoint_url=oscar.minio_endpoint,
        aws_access_key_id=oscar.minio_access_key,
        aws_secret_access_key=oscar.minio_secret_key,
        verify=oscar.minio_verify,
    )

    client.put_object(Bucket=oscar.service_name, Key=input_key, Body=content)
    await_oscar_job_completion(
        client,
        oscar.service_name,
        input_key,
        output_key,
        oscar=oscar,
    )
    return _read_object_bytes(client, oscar.service_name, output_key)

def get_oscar_job_status(
    oscar_uri: str,
    oscar_access_token: str,
    service_name: str,
    file_name_hash: str,
    timeout_seconds: int,
) -> str | None:
    response = requests.get(
        f"{oscar_uri.rstrip('/')}/system/logs/{service_name}?file={file_name_hash}",
        timeout=timeout_seconds,
        headers={"Authorization": f"Bearer {oscar_access_token}"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            "[OSCAR] returned an unexpected status code "
            f"{response.status_code} while checking job {file_name_hash}"
        )

    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("[OSCAR] returned an invalid job status response")

    jobs = payload.get("jobs", {})
    if not isinstance(jobs, dict):
        raise TypeError("[OSCAR] returned an invalid jobs collection")
    if not jobs:
        return None

    job = next(iter(jobs.values()))
    if not isinstance(job, dict):
        raise TypeError("[OSCAR] returned an invalid job record")
    status = job.get("status")
    if not isinstance(status, str):
        raise TypeError("[OSCAR] returned an invalid job status")
    return status


def await_oscar_job_completion(
    client,
    bucket: str,
    input_key: str,
    output_key: str,
    oscar: OSCAR,
) -> None:
    """Wait for Oscar to finish and verify that the S3 output object exists."""
    if oscar.timeout_seconds <= 0:
        raise ValueError("oscar.timeout_seconds must be greater than zero")
    if oscar.poll_seconds <= 0:
        raise ValueError("oscar.poll_seconds must be greater than zero")

    deadline = time.monotonic() + oscar.timeout_seconds
    filename_hash = filename_sha224(input_key)
    unknown_status_tolerance_checks = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        status = get_oscar_job_status(
            oscar.endpoint,
            oscar.access_token,
            oscar.service_name,
            filename_hash,
            timeout_seconds=min(1, remaining),
        )

        if status is None:
            unknown_status_tolerance_checks += 1
        else:
            normalized_status = status.casefold().strip()
            if normalized_status == "running":
                unknown_status_tolerance_checks = 0
            elif normalized_status == "succeeded":
                unknown_status_tolerance_checks = 0
                try:
                    client.head_object(Bucket=bucket, Key=output_key)
                    return
                except BotoCoreClientError as exc:
                    error = exc.response.get("Error", {})
                    code = str(error.get("Code", ""))
                    status_code = exc.response.get("ResponseMetadata", {}).get(
                        "HTTPStatusCode"
                    )
                    if status_code == 404 or code in {
                        "404",
                        "NoSuchKey",
                        "NotFound",
                    }:
                        raise RuntimeError(
                            "[OSCAR] job succeeded but the output object was not found"
                        ) from exc
                    raise
            elif normalized_status == "failed":
                raise RuntimeError("[OSCAR] job executed but failed")
            else:
                unknown_status_tolerance_checks += 1

        if unknown_status_tolerance_checks > 4:
            raise RuntimeError(
                "[OSCAR] unknown job status threshold exceeded. Job may not have been created or is not being tracked."
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(oscar.poll_seconds, remaining))

    raise TimeoutError(
        "[OSCAR] job did not complete within the expected time frame of "
        f"{oscar.timeout_seconds} seconds."
    )


def filename_sha224(filename: str) -> str:
    filename = filename.split("/", 1)[1] if "/" in filename else filename
    return hashlib.sha224(filename.encode("utf-8")).hexdigest()

def _doc_esxtraction_keys(filename: str | Path) -> tuple[str, str]:
    suffix = Path(filename).suffix
    hashed_name = hashlib.sha256(
        f"{filename}:{time.time_ns()}".encode()
    ).hexdigest()[:16]
    return f"input/{hashed_name}{suffix}", f"output/{hashed_name}.md"


def _read_object_bytes(client, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()