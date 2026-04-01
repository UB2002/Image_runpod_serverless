FROM runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /


# Start the worker
CMD ["python3", "-u", "handler.py"]
