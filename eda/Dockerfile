FROM python:3.7-slim

COPY . /app
WORKDIR /app
RUN python3 -m pip install -r requirements.txt
EXPOSE 8501 
ENTRYPOINT [ "streamlit", "run" ]
CMD ["recommender.py"]