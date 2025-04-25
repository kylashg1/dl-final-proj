import re
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config

TARGET_DIR = Path("../dl-final-proj/unprocessed_data")

# OpenCell dataset website
BUCKET = "czb-opencell"
PREFIX = "microscopy/raw/"

def download_from_OpenCell(limit=0):
    """
    dowloads cell images from OpenCell dataset
    limit: the max number of images this script download, for local downloading/testing purposes; a limit of 0 means download everything!!!
    """
    # can limit the amount of images downlaoded from OpenCell dataset for local testing purposes
    print(f"connecting to S3 bucket '{BUCKET}' to download 1 image per unique ENSG_id (up to limit:{limit})...")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    ensg_seen = set()
    count = 0
    ensg_regex = re.compile(r'(ENSG\d{11})')

    if not limit:  # catches limit=None or limit=0
        limit = float('inf')

    for pg in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in pg.get("Contents", []):
            key = obj["Key"]

            if not key.endswith("_proj.tif"):
                continue

            # attempt to extrac ENSG_id from filenames
            match = ensg_regex.search(key)
            if not match:
                continue

            ensg_id = match.group(1)
            if ensg_id in ensg_seen:
                continue

            # new ENSG_id -> download file
            filename = key.split("/")[-1]
            target_path = TARGET_DIR / filename
            s3.download_file(BUCKET, key, str(target_path))

            print(f"downloaded {ensg_id}: {filename}")
            ensg_seen.add(ensg_id)
            count += 1

            if count >= limit:
                print(f"downloaded {count} unique ENSG files.")
                return

    # print(f"downloading done!!! -> {count} unique ENSG_id cell images downloaded from OpenCell!!!.")
