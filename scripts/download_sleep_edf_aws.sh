#!/bin/bash

DESTINATION="data/raw/sleep-edfx"

if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found."
    echo "Install it with:"
    echo "  macOS: brew install awscli"
    echo "  Linux: pip install awscli"
    echo "  Or visit: https://aws.amazon.com/cli/"
    exit 1
fi

echo "Downloading Sleep-EDF dataset from AWS S3..."
echo "This may be faster than wget for large datasets."
echo "Destination: $(pwd)/$DESTINATION"
echo ""

mkdir -p "$DESTINATION"

aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ "$DESTINATION/1.0.0/" --cli-read-timeout 0

echo ""
echo "Download complete. Data should be in: $(pwd)/$DESTINATION/1.0.0/"
