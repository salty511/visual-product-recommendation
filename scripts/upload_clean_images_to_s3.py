#!/usr/bin/env python3
"""Upload all image files from data/clean to an S3 bucket."""

from __future__ import annotations

import argparse
import mimetypes
import os
from pathlib import Path
from typing import Iterable
from tqdm.auto import tqdm

import boto3
from botocore.exceptions import BotoCoreError, ClientError

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload image files from a local directory to an S3 bucket.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Destination S3 bucket name.",
    )
    parser.add_argument(
        "--source-dir",
        default="data/clean",
        help="Directory containing image files to upload (default: data/clean).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional S3 key prefix (example: clean-images/).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional AWS profile name.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Optional AWS region (overrides default config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded without performing upload.",
    )
    return parser.parse_args()


def iter_image_files(source_dir: Path) -> Iterable[Path]:
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def s3_key_for(path: Path, source_dir: Path, prefix: str) -> str:
    relative_path = path.relative_to(source_dir).as_posix()
    clean_prefix = prefix.strip("/")
    if clean_prefix:
        return f"{clean_prefix}/{relative_path}"
    return relative_path


def build_s3_client(profile: str | None, region: str | None):
    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.session.Session(**session_kwargs)

    client_kwargs = {}
    if region:
        client_kwargs["region_name"] = region
    return session.client("s3", **client_kwargs)


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}")
        return 1

    image_files = sorted(iter_image_files(source_dir))
    if not image_files:
        print(f"No image files found in {source_dir}")
        return 0

    print(f"Found {len(image_files)} image files in {source_dir}")

    if args.dry_run:
        for image_path in image_files:
            key = s3_key_for(image_path, source_dir, args.prefix)
            print(f"[DRY RUN] {image_path} -> s3://{args.bucket}/{key}")
        return 0

    s3_client = build_s3_client(args.profile, args.region)

    uploaded = 0
    failed = 0

    for image_path in tqdm(image_files):
        key = s3_key_for(image_path, source_dir, args.prefix)
        content_type, _ = mimetypes.guess_type(image_path.name)

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            s3_client.upload_file(
                Filename=os.fspath(image_path),
                Bucket=args.bucket,
                Key=key,
                ExtraArgs=extra_args,
            )
            uploaded += 1
        except (BotoCoreError, ClientError) as error:
            failed += 1
            print(f"Failed: {image_path} ({error})")

    print(f"Done. Uploaded: {uploaded}, Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
