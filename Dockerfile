# getting base ubuntu image
FROM ubuntu

MAINTAINER kmcleste pal <kmcleste@uncc.edu>

# runs at creation of image
RUN apt-get update -y
RUN apt-get install git -y
RUN apt-get install python3-pip -y
RUN apt-get install zip unzip
RUN git clone https://github.com/kmcleste/DSBA-6100.git

USER root
RUN cd DSBA-6100 && python3 -m pip install -r requirements.txt && unzip data.zip && streamlit run recommender.py


