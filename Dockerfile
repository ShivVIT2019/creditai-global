FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY agent_tab.py .
COPY agent/ ./agent/
COPY ml/ ./ml/

EXPOSE 8080

ENV PORT=8080

CMD ["python", "app.py"]
