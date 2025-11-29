# Базовый образ Python (легковесный)
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка системных зависимостей (libgomp1 нужен для CatBoost/LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и файлы модели
# Важно: model.cbm и features.json должны лежать рядом с Dockerfile при сборке
COPY app.py .
COPY model.cbm .
COPY features.json .

# Открываем порт 8000
EXPOSE 8000

# Команда запуска сервера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]