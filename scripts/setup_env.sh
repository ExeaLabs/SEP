#!/usr/bin/env bash
set -e

echo "=== Setting up SEP Environment ==="

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Environment Setup Complete ==="
echo "Run 'source .venv/bin/activate' before running scripts."
