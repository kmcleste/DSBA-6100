# getting base image
FROM ubuntu

LABEL kmcleste pal <kmcleste@uncc.edu>

WORKDIR /app

# runs at creation of image
RUN apt-get update -y
RUN apt-get install git -y
RUN apt-get install python3-pip -y
RUN apt-get install zip unzip

WORKDIR /app
RUN git clone https://github.com/kmcleste/DSBA-6100

WORKDIR /app/DSBA-6100
RUN unzip data.zip
RUN python3 -m pip install -r requirements.txt
CMD ["streamlit", "run", "recommender.py"]

EXPOSE 8501

