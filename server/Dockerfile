FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy all source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
