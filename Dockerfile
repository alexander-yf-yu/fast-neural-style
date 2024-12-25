FROM python:3.8

# Copy project files
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt && ./download_styling_models.sh

WORKDIR /data

ENTRYPOINT ["python", "/app/neural_style/neural_style.py"]
