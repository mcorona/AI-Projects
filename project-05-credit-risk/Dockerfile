# Serves app.py (the Streamlit dashboard) only -- not training, not
# evaluation. Uses requirements-app.txt, which is four packages: the
# dashboard recomputes every threshold, cost and disparity from the
# committed scored test split, so no scikit-learn, no joblib, no model
# file and no dataset need to be in the image.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app.py .
COPY output/reports/ output/reports/

EXPOSE 8501

# Everything the dashboard needs is baked in -- output/reports/ holds
# test_scores.csv.gz (201 KB, the 7,500 held-out predictions) plus the JSON
# reports. No volume mount, no dataset download:
#
#   docker build -t credit-decisions .
#   docker run -p 8501:8501 credit-decisions
#
# To retrain or re-evaluate, use the venv and requirements.txt instead;
# that path needs the 5 MB dataset, which downloads itself.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
