FROM python:3.10-slim as base

# Install system dependencies: ffmpeg is required for audio/video processing.
# You may optionally install Real‑ESRGAN if you need high quality 4K upscaling.
RUN apt-get update && \
    # Install FFmpeg and development libraries required by the PyAV package.
    # PyAV depends on the FFmpeg libraries to decode audio and video.  Without
    # the -dev packages, building the 'av' wheel (a dependency of faster‑whisper)
    # will fail with missing pkg-config files【529277485016824†L12-L23】.  We also
    # install pkg-config so that setup scripts can locate these libraries.
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

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