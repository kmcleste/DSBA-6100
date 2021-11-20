# Project BlockCluster

## What is it

BlockCluster is a real-time movie recommendation engine that uses content-based filtering. Our goal is to enable users to find movies that are relevant to those they already know and love. Users can access a lightweight web application created using Streamlit to run their recommendation queries.

## Getting started

1. Clone this github to your local computer by running either the following shell command or download/extract the [zip file](https://github.com/kmcleste/DSBA-6100/archive/refs/heads/main.zip) from github

    ```shell
    git clone https://github.com/kmcleste/DSBA-6100/recommender.git
    ```

2. Open the new github directory and install the required python packages

    ```shell
    cd ~/DSBA-6100
    python3 -m pip install -r requirements.txt
    ```

3. Extract ***data.zip*** and store the contents as such (you may delete the zip file after extraction):

    ```shell
    DSBA-6100 (folder)
    |   README.md
    |   recommender.py
    |
    |___data (folder)
        |   credits.csv
        |   keywords.csv
        |   movies_metadata.csv
    ```

4. Make sure you are in the DSBA-6100 directory and run the following command from your terminal of choice:

    ```shell
    streamlit run recommender.py
    ```

    To access the web application, go to <http://localhost:8501> in your web browser

---  

<br>

| Team Member   |       Email         |
| -----------   | --------------------|
| Kyle McLester | <kmcleste@uncc.edu> |
| Pruneet Inturi| <pinturi@uncc.edu>  |
| Pj Yoder      | <pyoder@uncc.edu>   |
| Luke Johnson  | <ljohn220@uncc.edu> |

<br>

University of North Carolina at Charlotte


Professor: [Dr. Zhang](https://belkcollege.charlotte.edu/directory/dongsong-zhang)
