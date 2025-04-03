FROM python:3.12

WORKDIR /app

# Копируем все файлы в контейнер
COPY . /app

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r new_req.txt

# Открываем порт 5000
EXPOSE 5000

# Запускаем Flask-приложение
CMD ["python", "summary_app.py"]