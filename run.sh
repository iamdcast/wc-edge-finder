#!/bin/bash
# Launch the WC Edge Finder dashboard.
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py
