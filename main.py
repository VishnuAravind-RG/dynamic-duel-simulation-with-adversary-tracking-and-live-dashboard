import streamlit as st
import numpy as np
import time
import pandas as pd
import pydeck as pdk
import random

from algorithms.astar_pathfinder import AStarPathfinder
from algorithms.hmm_predictor import HMMTrafficPredictor
from algorithms.fuzzy_logic import FuzzyUrgencyController
from algorithms.genetic_fleet import GeneticFleetOptimizer
from algorithms.bayesian_risk import BayesianRiskNet
from algorithms.minimax_adversary import MinimaxAdversary
from database.telemetry import TelemetryLogger

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="PSG Tech Autonomous Fleet AI", layout="wide")

def grid_to_gps(r, c, base_lat=11.0247, base_lon=77.0028, step=0.0015):
    return base_lat - (r * step), base_lon + (c * step)

# ---------------- SESSION INIT ---------------- #

if "initialized" not in st.session_state:

    st.session_state.logger = TelemetryLogger(batch_size=5)
    st.session_state.fuzzy = FuzzyUrgencyController()
    st.session_state.hmm = HMMTrafficPredictor(
        ['Clear', 'Congested'],
        ['Fast', 'Slow'],
        [0.8, 0.2],
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.9, 0.1], [0.2, 0.8]]
    )
    st.session_state.bayesian = BayesianRiskNet()

    tasks = [f"ORD-{i:03d}" for i in range(20)]
    vehicles = ["TRK-A", "TRK-B", "TRK-C"]
    ga = GeneticFleetOptimizer(tasks, vehicles)
    st.session_state.fleet_assignment = ga.optimize()

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

# ---------------- RANDOM GOAL ---------------- #

def generate_random_goal(grid, start):
    rows = len(grid)
    cols = len(grid[0])
    while True:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if grid[r][c] == 0 and (r, c) != start:
            return (r, c)

# ---------------- PATH EVALUATION ---------------- #

def evaluate_path(path):
    length = len(path)
    congestion = 0

    for i in range(length):
        if i % 3 == 0:
            congestion += 0.7
        else:
            congestion += 0.2

    avg_congestion = congestion / length
    cost = length + (avg_congestion * 10)

    return length, avg_congestion, cost

def generate_adversary_path(grid, start, goal):
    astar = AStarPathfinder(grid)
    base_path = astar.find_path(start, goal)

    if not base_path:
        return []

    adversary_path = base_path[:]

    # inject detours
    for _ in range(3):
        if len(adversary_path) < 3:
            break

        idx = random.randint(1, len(adversary_path)-2)
        r, c = adversary_path[idx]

        neighbors = [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]
        random.shuffle(neighbors)

        for nr, nc in neighbors:
            if 0 <= nr < 10 and 0 <= nc < 10 and grid[nr][nc] == 0:
                adversary_path.insert(idx, (nr, nc))
                break

    return adversary_path

# ---------------- UI ---------------- #

col_map, col_side = st.columns([2, 1])
map_placeholder = col_map.empty()
metrics_placeholder = col_side.empty()

if st.sidebar.button("DISPATCH TRK-A"):

    goal = generate_random_goal(grid, start)
    st.sidebar.success(f"New Destination: {goal}")

    astar = AStarPathfinder(grid)
    user_path = astar.find_path(start, goal)
    adversary_path = generate_adversary_path(grid, start, goal)

    if not user_path:
        st.error("No valid path found.")
        st.stop()

    total_user = len(user_path)
    total_adv = len(adversary_path)

    max_steps = max(total_user, total_adv)
    sleep_time = 0.6

    user_cum_cong = 0
    adv_cum_cong = 0

    start_time = time.time()

    for step in range(max_steps):

        # ---------------- USER STEP ----------------
        if step < total_user:
            ur, uc = user_path[step]
            user_lat, user_lon = grid_to_gps(ur, uc)
            user_current_cong = np.random.uniform(0.2, 0.8)
            user_cum_cong += user_current_cong
            user_avg_cong = user_cum_cong / (step + 1)
        else:
            user_current_cong = 0
            user_avg_cong = user_cum_cong / total_user

        # ---------------- ADVERSARY STEP ----------------
        if step < total_adv:
            ar, ac = adversary_path[step]
            adv_lat, adv_lon = grid_to_gps(ar, ac)
            adv_current_cong = np.random.uniform(0.3, 0.9)
            adv_cum_cong += adv_current_cong
            adv_avg_cong = adv_cum_cong / (step + 1)
        else:
            adv_current_cong = 0
            adv_avg_cong = adv_cum_cong / total_adv

        elapsed = time.time() - start_time

        # ---------------- MAP ----------------
        user_latlon = [grid_to_gps(x, y) for x, y in user_path]
        adv_latlon = [grid_to_gps(x, y) for x, y in adversary_path]

        layers = [
            # Adversary path
            pdk.Layer(
                "PathLayer",
                pd.DataFrame({
                    "path": [[[lon2, lat2] for lat2, lon2 in adv_latlon]]
                }),
                get_path="path",
                get_color=[255, 0, 255],
                get_width=4,
            ),
            # User path
            pdk.Layer(
                "PathLayer",
                pd.DataFrame({
                    "path": [[[lon2, lat2] for lat2, lon2 in user_latlon]]
                }),
                get_path="path",
                get_color=[0, 150, 255],
                get_width=6,
            ),
            # User truck
            pdk.Layer(
                "ScatterplotLayer",
                pd.DataFrame({"lat":[user_lat], "lon":[user_lon]}),
                get_position='[lon, lat]',
                get_radius=120,
                get_fill_color=[255,50,50],
            ),
            # Adversary truck
            pdk.Layer(
                "ScatterplotLayer",
                pd.DataFrame({"lat":[adv_lat], "lon":[adv_lon]}),
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

        # ---------------- DASHBOARD ----------------
        with metrics_placeholder.container():

            st.markdown("## 🚀 Live Duel Dashboard")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🚛 Your Fleet")
                st.metric("Steps Completed", min(step+1, total_user))
                st.metric("Avg Congestion", f"{user_avg_cong:.2f}")
                st.metric("Current Congestion", f"{user_current_cong:.2f}")
                st.progress(min((step+1)/total_user,1.0))

            with col2:
                st.markdown("### 🧠 Adversary")
                st.metric("Steps Completed", min(step+1, total_adv))
                st.metric("Avg Congestion", f"{adv_avg_cong:.2f}")
                st.metric("Current Congestion", f"{adv_current_cong:.2f}")
                st.progress(min((step+1)/total_adv,1.0))

            st.markdown("---")

            # Live winner indicator
            if user_avg_cong < adv_avg_cong:
                st.success("🏆 Advantage: YOUR PATH")
            elif user_avg_cong > adv_avg_cong:
                st.error("⚠ Advantage: ADVERSARY")
            else:
                st.info("⚖ Currently Balanced")

        time.sleep(sleep_time)

    st.success("Simulation Completed")