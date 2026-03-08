#!/bin/bash

TOKEN="29983-7eacrau1xZ85mzhbbHp1"

if ! command -v nsrr &> /dev/null; then
    echo "Error: nsrr CLI tool not found."
    echo "Install it with: pip install nsrr"
    exit 1
fi

echo "Downloading SHHS dataset with token..."
nsrr download shhs --shallow --token "$TOKEN"

echo "Download complete. Data should be in the current directory."
