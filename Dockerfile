# getting base ubuntu image
FROM ubuntu

LABEL kmcleste pal <kmcleste@uncc.edu>

WORKDIR /app

# runs at creation of image
RUN apt-get update -y
RUN apt-get install git -y
RUN apt-get install python3-pip -y
RUN apt-get install zip unzip
RUN git clone https://github.com/kmcleste/DSBA-6100.git
RUN cd DSBA-6100 && python3 -m pip install -r requirements.txt && unzip data.zip
WORKDIR /app/DSBA-6100
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["recommender.py"]