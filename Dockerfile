# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir openai pydantic python-dotenv openenv-core

# Ensure Python can find env.py from inside the server folder
ENV PYTHONPATH="/app"

# Copy the root files
COPY env.py /app/
COPY inference.py /app/
COPY openenv.yaml /app/
COPY pyproject.toml /app/
COPY uv.lock /app/

RUN mkdir -p /app/server
COPY server/app.py /app/server/app.py

# RUN COMMAND
CMD ["sh", "-c", "python server/app.py"]