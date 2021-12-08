# Project BlockCluster

## What is it

BlockCluster is a real-time movie recommendation engine that uses collaborative filtering. Our goal is to enable users to find movies that are relevant to those they already know and love. Users can access a lightweight web application created using Streamlit to run their recommendation queries.

## Getting started

1. Clone this github to your local computer by running the following shell command

    ```bash
    git clone https://github.com/kmcleste/DSBA-6100.git
    ```

2. Open the new github directory and install the required python packages

    ```bash
    cd DSBA-6100/recommender_v1.2

    virtualenv ./venv
    source ./venv/bin/activate

    python3 -m pip install -r requirements.txt
    ```


3. Make sure you are in the DSBA-6100/recommender_v1.2 directory and run the following command:

    ```bash
    streamlit run recommender.py
    # to change port, use streamlit run recommender.py --server.port $PORT
    # where $PORT is your desired port
    ```

    To access the web application, go to <http://localhost:8501> in your web browser

## Docker (optional)

1. Ensure docker and docker.io are installed. You may have to run the following command to start docker service:

    ```bash
    systemctl start docker
    ```


2. Pull the latest docker container:

    ```bash
    docker pull kam2897/blockcluster:recommender
    ```

3. Run the docker container:

    ```bash
    docker run -d -p 8501:8501 kam2897/blockcluster:recommender
    ```

4. Access the web application at <http://localhost:8501> in your web browser

## Building Custom Docker Image (optional)

1. Create Dockerfile with the following contents:

    ```bash
    FROM python:3.7-slim

    COPY . /app
    WORKDIR /app
    RUN python3 -m pip install -r requirements.txt
    EXPOSE $PORT    # where $PORT is your desired port 
    ENTRYPOINT [ "streamlit", "run" ]
    CMD ["recommender.py", "--server.port", "$PORT"]
    ```

2. Build your docker image with:

    ```bash
    docker build -t $docker_hub_username/$project_name:$tag
    ```

3. Run your new docker container:

    ```bash
    docker run -d -p $PORT:$PORT $docker_hub_username/$project_name:$tag
    ```

4. Push to docker hub (optional):

    ```bash
    docker push $docker_hub_username/$project_name:$tag
    ```


---  

<br>

| Team Member   |       Email         |
| -----------   | --------------------|
| Kyle McLester | <kmcleste@uncc.edu> |
| Praneeth Inturi| <pinturi@uncc.edu>  |
| PJ Yoder      | <pyoder@uncc.edu>   |
| Luke Johnson  | <ljohn220@uncc.edu> |

<br>

University of North Carolina at Charlotte


Professor: [Dr. Zhang](https://belkcollege.charlotte.edu/directory/dongsong-zhang)
