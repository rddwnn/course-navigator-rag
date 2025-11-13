FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        python3-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml ./

RUN uv --version && uv sync --no-dev

RUN mkdir -p /app/data

COPY src/ ./src/

CMD ["uv", "run", "python", "-m", "course_navigator_rag.bot"]
