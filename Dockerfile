FROM python:3.10-slim as base

# Install system dependencies: ffmpeg is required for audio/video processing.
# You may optionally install Real‑ESRGAN if you need high quality 4K upscaling.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source
COPY . .

# Provide an entrypoint. The bot token should be provided via environment
# variable BOT_TOKEN or set in bot.py. To override, specify
# `-e BOT_TOKEN=your_token` when running the container.
ENV PYTHONUNBUFFERED=1
CMD ["python", "bot.py"]