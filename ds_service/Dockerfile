# Используем python:3.10-slim — это золотая середина (Debian-based, но маленький)
FROM python:3.10-slim

# 1. Переменные окружения для оптимизации Python в контейнере
# PYTHONDONTWRITEBYTECODE=1: не создавать .pyc файлы (экономит место)
# PYTHONUNBUFFERED=1: логи сразу летят в консоль (удобно для отладки)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 2. Установка системных библиотек
# Объединяем команды в один RUN, чтобы не плодить слои образа
# libgomp1 нужен для CatBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 3. Установка зависимостей (КЭШИРУЕМЫЙ СЛОЙ)
# Мы копируем ТОЛЬКО requirements.txt перед всем кодом.
# Если ты поменяешь код в app.py, но не тронешь requirements,
# Docker НЕ БУДЕТ переустанавливать библиотеки. Сборка займет 1 секунду.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Копируем код приложения и файлы модели
# .dockerignore (см. ниже) поможет не копировать мусор
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]