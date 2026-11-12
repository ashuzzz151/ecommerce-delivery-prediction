FROM python:3.14-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the web service and trained model
COPY serve.py model.bin ./

# Expose the FastAPI port
EXPOSE 9696

# Launch the web service
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-9696}"]