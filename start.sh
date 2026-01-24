#!/bin/bash

# Start FastAPI in background
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to start
sleep 5

# Start Streamlit in foreground
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true