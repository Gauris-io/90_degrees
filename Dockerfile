# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install required Python packages directly
RUN pip install --no-cache-dir openai pydantic

# Copy your local project files into the Docker container
COPY env.py /app/
COPY inference.py /app/
COPY openenv.yaml /app/
COPY pyproject.py /app/
COPY uv.lock /app/
RUN mkdir -p /app/server
COPY server/app.py /app/server/app.py
# The command that will run when the automated test starts
CMD ["sh", "-c", "python inference.py & python server/app.py"]