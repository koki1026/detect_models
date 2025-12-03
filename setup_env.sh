#!/bin/bash
set -e

echo "🔧 Creating venv in .venv ..."
python3 -m venv .venv

echo "🟢 Activating venv ..."
source .venv/bin/activate

echo "📦 Installing dependencies ..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "✨ Environment setup complete!"
echo "To activate environment again: source .venv/bin/activate"