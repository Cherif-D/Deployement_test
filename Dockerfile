# Image légère + Python 3.12
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code + modèle
COPY . .

# 3) Exposition port Streamlit
EXPOSE 8501

# 4) Lancement de l'app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
