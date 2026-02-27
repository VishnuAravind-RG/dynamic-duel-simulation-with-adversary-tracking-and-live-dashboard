import streamlit as st
import numpy as np
import time
import pandas as pd
import pydeck as pdk
import random

from algorithms.astar_pathfinder import AStarPathfinder
from algorithms.hmm_predictor import HMMTrafficPredictor
from algorithms.bayesian_risk import BayesianRiskNet
from algorithms.minimax_adversary import MinimaxAdversary

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="Autonomous Fleet Duel Intelligence", layout="wide")

def grid_to_gps(r, c, base_lat=11.0247, base_lon=77.0028, step=0.0015):
    return base_lat - (r * step), base_lon + (c * step)

# ---------------- SESSION INIT ---------------- #

if "initialized" not in st.session_state:
    st.session_state.hmm = HMMTrafficPredictor(
        ['Clear', 'Congested'],
        ['Fast', 'Slow'],
        [0.8, 0.2],
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.9, 0.1], [0.2, 0.8]]
    )
    st.session_state.bayesian = BayesianRiskNet()
    st.session_state.initialized = True

# ---------------- GRID ---------------- #

grid = [
    [0,0,0,1,0,0,0,0,0,0],
    [0,1,0,1,0,1,1,0,1,0],
    [0,1,0,0,0,0,1,0,0,0],
    [0,0,0,1,1,0,0,0,1,0],
    [1,1,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,1,1,0,1,0],
    [0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,1,0],
    [1,1,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0]
]

start = (0, 0)

def generate_random_goal(grid, start):
    while True:
        r = random.randint(0, 9)
        c = random.randint(0, 9)
        if grid[r][c] == 0 and (r, c) != start:
            return (r, c)

goal = generate_random_goal(grid, start)

# ---------------- PATH GENERATION ---------------- #

astar = AStarPathfinder(grid)
user_path = astar.find_path(start, goal)

def generate_adversary_path(base_path):
    path = base_path.copy()
    for _ in range(3):
        if len(path) > 3:
            idx = random.randint(1, len(path)-2)
            r, c = path[idx]
            neighbors = [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]
            random.shuffle(neighbors)
            for nr, nc in neighbors:
                if 0 <= nr < 10 and 0 <= nc < 10 and grid[nr][nc] == 0:
                    path.insert(idx,(nr,nc))
                    break
    return path

adversary_path = generate_adversary_path(user_path)

# ---------------- PRE-MISSION DUEL PANEL ---------------- #

st.markdown("## 🧠 Pre-Mission Duel Intelligence")

if not user_path:
    st.error("No route available.")
    st.stop()

user_length = len(user_path)
adv_length = len(adversary_path)

# HMM Prediction
obs_seq = ["Fast","Slow","Fast"]
traffic_probs = st.session_state.hmm.forward_algorithm(obs_seq)
expected_congestion = traffic_probs[1]

# Bayesian Risk
iteration = random.randint(1,10)
is_raining = st.session_state.bayesian.get_weather_state(iteration)
delay_user = st.session_state.bayesian.infer_delay_probability(
    is_raining, expected_congestion > 0.5
)

delay_adv = min(1.0, delay_user + 0.15)  # adversary assumed slightly worse

success_user = max(0.0, 1 - (delay_user * expected_congestion))
success_adv = max(0.0, 1 - (delay_adv * expected_congestion))

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚛 Your Forecast")
    st.metric("Path Length", user_length)
    st.metric("Delay Probability", f"{delay_user:.2f}")
    st.metric("Success Probability", f"{success_user:.2f}")

with col2:
    st.markdown("### 🧠 Adversary Forecast")
    st.metric("Path Length", adv_length)
    st.metric("Delay Probability", f"{delay_adv:.2f}")
    st.metric("Success Probability", f"{success_adv:.2f}")

if success_user > success_adv:
    st.success("🏆 Pre-Mission Advantage: YOU")
elif success_user < success_adv:
    st.error("⚠ Pre-Mission Advantage: ADVERSARY")
else:
    st.info("⚖ Balanced Duel")

st.markdown("---")

# ---------------- LIVE DUEL ---------------- #

if st.button("🚀 START DUEL"):

    map_placeholder = st.empty()
    metrics_placeholder = st.empty()

    max_steps = max(len(user_path), len(adversary_path))
    user_cum = 0
    adv_cum = 0

    for step in range(max_steps):

        if step < len(user_path):
            ur, uc = user_path[step]
            user_lat, user_lon = grid_to_gps(ur, uc)
            user_cong = np.random.uniform(0.2,0.7)
            user_cum += user_cong
            user_avg = user_cum/(step+1)
        else:
            user_avg = user_cum/len(user_path)

        if step < len(adversary_path):
            ar, ac = adversary_path[step]
            adv_lat, adv_lon = grid_to_gps(ar, ac)
            adv_cong = np.random.uniform(0.3,0.9)
            adv_cum += adv_cong
            adv_avg = adv_cum/(step+1)
        else:
            adv_avg = adv_cum/len(adversary_path)

        layers = [
            pdk.Layer(
                "PathLayer",
                pd.DataFrame({"path":[[[grid_to_gps(x,y)[1],grid_to_gps(x,y)[0]] for x,y in user_path]]}),
                get_path="path",
                get_color=[0,150,255],
                get_width=6,
            ),
            pdk.Layer(
                "PathLayer",
                pd.DataFrame({"path":[[[grid_to_gps(x,y)[1],grid_to_gps(x,y)[0]] for x,y in adversary_path]]}),
                get_path="path",
                get_color=[255,0,255],
                get_width=4,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                pd.DataFrame({"lat":[user_lat],"lon":[user_lon]}),
                get_position='[lon, lat]',
                get_radius=120,
                get_fill_color=[255,50,50],
            ),
            pdk.Layer(
                "ScatterplotLayer",
                pd.DataFrame({"lat":[adv_lat],"lon":[adv_lon]}),
                get_position='[lon, lat]',
                get_radius=120,
                get_fill_color=[255,0,255],
            )
        ]

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=user_lat,
                longitude=user_lon,
                zoom=15,
            ),
        )

        map_placeholder.pydeck_chart(deck)

        with metrics_placeholder.container():
            st.markdown("### 🔥 Live Duel Metrics")
            colA, colB = st.columns(2)

            with colA:
                st.metric("Your Avg Congestion", f"{user_avg:.2f}")

            with colB:
                st.metric("Adversary Avg Congestion", f"{adv_avg:.2f}")

            if user_avg < adv_avg:
                st.success("🏆 You Leading")
            elif user_avg > adv_avg:
                st.error("⚠ Adversary Leading")
            else:
                st.info("⚖ Equal")

        time.sleep(0.6)

    st.success("Duel Completed")