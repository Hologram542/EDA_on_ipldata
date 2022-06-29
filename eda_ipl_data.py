import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import streamlit as st


pickle_a = open("C:\\Users\\Ravikiran\\Downloads\\eda_ipl_data.h5", "rb")
regressor = pickle.load(pickle_a)  # our model


def predict_chance(inning, over, total_runs, player_dismissed, innings_wickets, innings_score, score_target, remaining_target, run_rate, required_run_rate, runrate_diff, is_batting_team):

    test = np.array([[inning, over, total_runs, player_dismissed, innings_wickets, innings_score, score_target,
                     remaining_target, run_rate, required_run_rate, runrate_diff, is_batting_team]])
    test = xgb.DMatrix(test)

    prediction = regressor.predict(test)  # predictions using our model
    return prediction


def main():
    st.title("Win prediction APP using ML")  # simple title for the app
    html_temp = """
        <div>
        <h2>IPL Win Prediction ML app</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)  # a simple html
    inning = st.radio("Innnigs", [1, 2])

    over = st.slider("Over", 0, 20)

    total_runs = st.slider("Total runs scored in this over", 0, 42)

    innings_score = st.slider("Total runs scored this innings", 0, 300)

    player_dismissed = st.radio("Is this a wicket ball ?", ["Yes", "No"])

    if player_dismissed == "Yes":
        player_dismissed = 1
    else:
        player_dismissed = 0

    innings_wickets = st.slider("Total wickets fallen", 0, 10)

    run_rate = innings_score/over

    if inning == 1:
        score_target = -1
        remaining_target = -1
        required_run_rate = -1
        runrate_diff = -1
    else:
        score_target = st.slider("Target", 0, 300)
        remaining_target = score_target - innings_score
        required_run_rate = remaining_target/(20 - over)
        runrate_diff = run_rate - required_run_rate

    is_batting_team = st.radio(
        "Which team's probability do you want to predict ?", ["Batting", "Bowling"])
    if is_batting_team == "Batting":
        is_batting_team = 1
    else:
        is_batting_team = 0

    result = 0
    if st.button("Predict"):
        # result will be displayed if button is pressed
        result = predict_chance(inning, over, total_runs, player_dismissed, innings_wickets, innings_score, score_target,
                                remaining_target, run_rate, required_run_rate, runrate_diff, is_batting_team)
    if is_batting_team == 1:
        st.success("The chance of this team winning is {} ".format(1 - result))
    else:
        st.success("The chance of this team winning is {} ".format(result))


if __name__ == '__main__':
    main()
