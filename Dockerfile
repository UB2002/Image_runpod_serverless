FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

WORKDIR /

# --break-system-packages required: Python 3.12 on Debian enforces PEP 668
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy handler
COPY handler.py /

# Start the worker
CMD ["python3", "-u", "handler.py"]