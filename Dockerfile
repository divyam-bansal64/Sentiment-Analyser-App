FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root system user for security
RUN useradd -m -u 1000 capstone_user && \
    chown -R capstone_user:capstone_user /app

# Copy requirement file first for layer caching
COPY flask_app/requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Download NLTK datasets into a globally accessible directory for capstone_user
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet

# Copy rest of application code and model artifacts with non-root ownership
COPY --chown=capstone_user:capstone_user flask_app/ /app/
COPY --chown=capstone_user:capstone_user models/vectorizer.pkl /app/models/vectorizer.pkl

# Switch to non-root user
USER capstone_user

EXPOSE 5000

# Local deployment (active for local verification)
# CMD ["python", "app.py"]

# Production deployment (uncomment for EKS / Gunicorn in production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]