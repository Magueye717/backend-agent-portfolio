# build.sh
#!/bin/bash
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Running ingestion script to populate ChromaDB..."
python ingest.py

echo "Build complete."