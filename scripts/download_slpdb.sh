#!/bin/bash

DATA_DIR="data/raw"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit 1

echo "Downloading MIT-BIH Polysomnographic Database (slpdb) from PhysioNet..."
echo "This is a small dataset (~<1GB) but may take a few minutes."

wget -r -N -c -np -A "dat,hea,st" -nH --cut-dirs=2 https://physionet.org/files/slpdb/1.0.0/

echo "Download complete. Data should be in: $(pwd)/slpdb/1.0.0/"
