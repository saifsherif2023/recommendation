#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start Gunicorn with the PORT environment variable
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers 4 --threads 2 app:app 