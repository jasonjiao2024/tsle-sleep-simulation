#!/bin/bash

DATA_DIR="data/raw"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading Sleep-EDF dataset from PhysioNet..."
echo "This may take a while (several GB of data)..."

wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/

echo "Download complete. Data should be in: $(pwd)/sleep-edfx/1.0.0/"
