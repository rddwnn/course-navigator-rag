FROM python:3.11-slim

# Чуть-чуть системных пакетов: curl для установки uv, build-essential для возможной сборки колёс
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv (без pip), добавляем в PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Копируем описание проекта (зависимости)
COPY pyproject.toml ./

# Синхронизируем зависимости (создаст окружение под uv)
# --no-dev чтобы не тянуть лишнее, если у тебя будут dev-зависимости
RUN uv sync --no-dev

# Создаём директорию для данных курсов (будет примонтирована с хоста)
RUN mkdir -p /app/data

# Копируем исходники
COPY src/ ./src/

# По умолчанию запускаем Telegram-бота через uv
CMD ["uv", "run", "python", "-m", "course_navigator_rag.bot"]
