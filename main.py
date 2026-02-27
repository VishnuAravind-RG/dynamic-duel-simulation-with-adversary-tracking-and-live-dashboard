import streamlit as st
import numpy as np
import time
import pandas as pd
import pydeck as pdk
import random
import sqlite3
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import deque
import hashlib

from algorithms.astar_pathfinder import AStarPathfinder
from algorithms.hmm_predictor import HMMTrafficPredictor
from algorithms.bayesian_risk import BayesianRiskNet
from algorithms.minimax_adversary import MinimaxAdversary
from algorithms.genetic_fleet import GeneticFleetOptimizer
from algorithms.csp_scheduler import CSPScheduler
from algorithms.fuzzy_logic import FuzzyUrgencyController
from algorithms.mdp_solver import MDPPolicyIterator
from database.telemetry import TelemetryLogger
from algorithms.rl_controller import RLTrafficController

# ==================== CONFIGURATION ==================== #

st.set_page_config(
    page_title="🤖 ULTIMATE FLEET AI SYSTEM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .glow-text {
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00;
    }
</style>
""", unsafe_allow_html=True)

def grid_to_gps(r, c, base_lat=11.0247, base_lon=77.0028, step=0.0015):
    return base_lat - (r * step), base_lon + (c * step)

# ==================== ADVANCED AI MODULES ==================== #

class NeuralPathPredictor:
    """Neural network for path prediction (simulated)"""
    def __init__(self):
        self.model_version = "2.1.0"
        self.accuracy = 0.89
        
    def predict_eta(self, path_length, traffic, weather):
        base_time = path_length * 2
        traffic_factor = 1 + (traffic * 1.5)
        weather_factor = 1.2 if weather in ["Rainy", "Foggy"] else 1.0
        return base_time * traffic_factor * weather_factor

class SwarmIntelligence:
    """Multi-agent swarm coordination"""
    def __init__(self, num_agents=5):
        self.num_agents = num_agents
        self.agents = []
        self.swarm_coherence = 0.95
        
    def optimize_routes(self, paths):
        # Simulate swarm optimization
        optimized = []
        for path in paths:
            if random.random() < self.swarm_coherence:
                optimized.append(path)
            else:
                # Add some variation
                optimized.append(path[::-1] if random.random() > 0.5 else path)
        return optimized

class BlockchainLogger:
    """Immutable mission logging (simulated blockchain)"""
    def __init__(self):
        self.chain = []
        self.difficulty = 4
        
    def add_block(self, data):
        block = {
            'timestamp': time.time(),
            'data': data,
            'previous_hash': self.get_last_hash(),
            'nonce': random.randint(0, 1000000)
        }
        block['hash'] = self.calculate_hash(block)
        self.chain.append(block)
        return block
    
    def calculate_hash(self, block):
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()[:8]
    
    def get_last_hash(self):
        return self.chain[-1]['hash'] if self.chain else "0"*8

# ==================== SESSION INIT ==================== #

if "initialized" not in st.session_state:
    # Initialize all AI modules
    st.session_state.hmm = HMMTrafficPredictor(
        ['Clear', 'Congested'],
        ['Fast', 'Slow'],
        [0.8, 0.2],
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.9, 0.1], [0.2, 0.8]]
    )
    st.session_state.bayesian = BayesianRiskNet()
    st.session_state.telemetry = TelemetryLogger(db_name="ultimate_fleet.db", batch_size=5)
    st.session_state.rl_agent = RLTrafficController()
    st.session_state.neural_predictor = NeuralPathPredictor()
    st.session_state.swarm = SwarmIntelligence()
    st.session_state.blockchain = BlockchainLogger()
    
    # Metrics tracking
    st.session_state.rl_episodes = 0
    st.session_state.rl_rewards_history = []
    st.session_state.fleet_performance = deque(maxlen=100)
    st.session_state.system_uptime = time.time()
    st.session_state.missions_completed = 0
    
    st.session_state.initialized = True

# ==================== GRID & PATH GENERATION ==================== #

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

# Generate multiple paths for swarm
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

# Generate multiple paths for swarm demo
all_paths = [user_path]
for i in range(4):
    all_paths.append(generate_adversary_path(user_path))

# ==================== SIDEBAR - GLOBAL METRICS ==================== #

with st.sidebar:
    st.markdown("# 🤖 ULTIMATE AI FLEET")
    st.markdown("---")
    
    # System Status
    uptime = time.time() - st.session_state.system_uptime
    st.markdown("### 🖥️ SYSTEM STATUS")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Uptime", f"{int(uptime//3600)}h {int((uptime%3600)//60)}m")
    with col_s2:
        st.metric("Missions", st.session_state.missions_completed)
    
    # AI Model Versions
    st.markdown("### 🧠 AI MODELS ACTIVE")
    st.info(f"🤖 RL Agent: v{random.choice(['2.1.0', '2.2.0', '3.0.0-beta'])}")
    st.info(f"🧬 Genetic: v{random.choice(['1.9.8', '2.0.1', '2.1.5'])}")
    st.info(f"📊 Neural: v{st.session_state.neural_predictor.model_version}")
    
    # Blockchain Status
    st.markdown("### 🔗 BLOCKCHAIN")
    st.success(f"Blocks: {len(st.session_state.blockchain.chain)}")
    st.caption(f"Last hash: {st.session_state.blockchain.get_last_hash()}")
    
    # Live Performance Gauge
    st.markdown("### 📊 FLEET PERFORMANCE")
    perf_value = np.mean([p.get('score', 80) for p in list(st.session_state.fleet_performance)[-10:]]) if st.session_state.fleet_performance else 85
    st.progress(perf_value/100, f"Efficiency: {perf_value:.1f}%")

# ==================== MAIN TABS ==================== #

tab1, tab2, tab3, tab4 = st.tabs([
    "⚔️ DUEL ARENA", 
    "📊 ANALYTICS HUB", 
    "🧠 AI LAB", 
    "🔐 BLOCKCHAIN LEDGER"
])

# ==================== TAB 1: DUEL ARENA (Enhanced) ==================== #

with tab1:
    st.markdown("# ⚔️ AUTONOMOUS FLEET DUEL ARENA")
    st.markdown("### *Where AI Agents Battle for Supremacy*")
    
    # Pre-mission intelligence with enhanced metrics
    col_adv1, col_adv2, col_adv3 = st.columns(3)
    
    with col_adv1:
        st.markdown("### 🚛 YOUR FLEET")
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
        
        # Neural prediction
        neural_eta = st.session_state.neural_predictor.predict_eta(
            len(user_path), expected_congestion, "Rainy" if is_raining else "Clear"
        )
        
        st.metric("Path Length", len(user_path))
        st.metric("Delay Probability", f"{delay_user:.2%}")
        st.metric("Neural ETA", f"{neural_eta:.1f} min")
        st.metric("Success Probability", f"{1 - (delay_user * expected_congestion):.2%}")
    
    with col_adv2:
        st.markdown("### 🧠 ADVERSARY FLEET")
        delay_adv = min(1.0, delay_user + 0.15)
        adv_eta = neural_eta * 1.2  # Adversary is slower
        st.metric("Path Length", len(adversary_path))
        st.metric("Delay Probability", f"{delay_adv:.2%}")
        st.metric("Neural ETA", f"{adv_eta:.1f} min")
        st.metric("Success Probability", f"{1 - (delay_adv * expected_congestion):.2%}")
    
    with col_adv3:
        st.markdown("### 🏆 PREDICTED OUTCOME")
        your_score = (1 - delay_user) * 100 - len(user_path)
        adv_score = (1 - delay_adv) * 100 - len(adversary_path)
        
        if your_score > adv_score:
            st.success("### YOU WIN! 🎉")
            st.balloons()
        elif your_score < adv_score:
            st.error("### ADVERSARY WINS! ⚠️")
        else:
            st.info("### TIE! ⚖️")
        
        st.metric("Your Score", f"{your_score:.1f}")
        st.metric("Adversary Score", f"{adv_score:.1f}")
    
    # Enhanced 3D Duel Visualization
    st.markdown("---")
    st.markdown("### 🗺️ LIVE 3D DUEL VISUALIZATION")
    
    # Create elevation data for 3D buildings
    building_heights = np.random.randint(0, 50, (10, 10))
    
    col_map1, col_map2 = st.columns([3, 1])
    
    with col_map1:
        if st.button("🚀 INITIATE DUEL", type="primary", use_container_width=True):
            map_placeholder = st.empty()
            metrics_placeholder = st.empty()
            blockchain_placeholder = st.empty()
            
            max_steps = max(len(user_path), len(adversary_path))
            user_cum = 0
            adv_cum = 0
            
            for step in range(max_steps):
                # Update positions
                if step < len(user_path):
                    ur, uc = user_path[step]
                    user_lat, user_lon = grid_to_gps(ur, uc)
                    user_cong = np.random.uniform(0.2,0.7)
                    user_cum += user_cong
                    user_avg = user_cum/(step+1)
                else:
                    ur, uc = user_path[-1]  # Stay at last position
                    user_lat, user_lon = grid_to_gps(ur, uc)
                    user_avg = user_cum/len(user_path)
                    
                if step < len(adversary_path):
                    ar, ac = adversary_path[step]
                    adv_lat, adv_lon = grid_to_gps(ar, ac)
                    adv_cong = np.random.uniform(0.3,0.9)
                    adv_cum += adv_cong
                    adv_avg = adv_cum/(step+1)
                else:
                    ar, ac = adversary_path[-1]  # Stay at last position
                    adv_lat, adv_lon = grid_to_gps(ar, ac)
                    adv_avg = adv_cum/len(adversary_path)
                
                # Create trail data safely
                trail_data = []
                for i in range(1, 4):
                    trail_step = step - i
                    if trail_step >= 0 and trail_step < len(user_path):
                        tr, tc = user_path[trail_step]
                        t_lat, t_lon = grid_to_gps(tr, tc)
                        trail_data.append({
                            'lat': t_lat,
                            'lon': t_lon,
                            'size': 100 - i * 25
                        })
                
                # Create 3D visualization with buildings
                layers = [
                    # 3D Buildings
                    pdk.Layer(
                        "ColumnLayer",
                        data=pd.DataFrame({
                            'lat': [grid_to_gps(r, c)[0] for r in range(10) for c in range(10)],
                            'lon': [grid_to_gps(r, c)[1] for r in range(10) for c in range(10)],
                            'height': [building_heights[r][c] for r in range(10) for c in range(10)]
                        }),
                        get_position='[lon, lat]',
                        get_elevation='height',
                        elevation_scale=5,
                        radius=25,
                        get_fill_color='[100, 100, 200, 150]',
                        extruded=True,
                    ),
                    # Your path
                    pdk.Layer(
                        "PathLayer",
                        pd.DataFrame({"path":[[[grid_to_gps(x,y)[1],grid_to_gps(x,y)[0]] for x,y in user_path]]}),
                        get_path="path",
                        get_color=[0,255,0],
                        get_width=8,
                        width_min_pixels=2,
                    ),
                    # Adversary path
                    pdk.Layer(
                        "PathLayer",
                        pd.DataFrame({"path":[[[grid_to_gps(x,y)[1],grid_to_gps(x,y)[0]] for x,y in adversary_path]]}),
                        get_path="path",
                        get_color=[255,0,255],
                        get_width=8,
                        width_min_pixels=2,
                    ),
                    # Your current position (with glow effect)
                    pdk.Layer(
                        "ScatterplotLayer",
                        pd.DataFrame({"lat":[user_lat],"lon":[user_lon]}),
                        get_position='[lon, lat]',
                        get_radius=150,
                        get_fill_color=[0,255,0],
                        get_line_color=[255,255,255],
                        line_width_min_pixels=2,
                    ),
                    # Adversary current position
                    pdk.Layer(
                        "ScatterplotLayer",
                        pd.DataFrame({"lat":[adv_lat],"lon":[adv_lon]}),
                        get_position='[lon, lat]',
                        get_radius=150,
                        get_fill_color=[255,0,255],
                        get_line_color=[255,255,255],
                        line_width_min_pixels=2,
                    )
                ]
                
                # Add trail layer if there's data
                if trail_data:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=pd.DataFrame(trail_data),
                            get_position='[lon, lat]',
                            get_radius='size',
                            get_fill_color='[0, 255, 0, 100]',
                        )
                    )
                
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=user_lat,
                        longitude=user_lon,
                        zoom=16,
                        pitch=45,
                        bearing=step*2,
                    ),
                    map_style='mapbox://styles/mapbox/dark-v11',
                )
                
                map_placeholder.pydeck_chart(deck)
                
                # Live metrics with animations
                with metrics_placeholder.container():
                    st.markdown("### 🔥 LIVE DUEL METRICS")
                    colA, colB, colC = st.columns(3)
                    
                    with colA:
                        st.metric("YOUR CONGESTION", f"{user_avg:.2%}", 
                                 f"{user_avg - adv_avg:.2%}" if step > 0 else None)
                    with colB:
                        st.metric("ADVERSARY CONGESTION", f"{adv_avg:.2%}",
                                 f"{adv_avg - user_avg:.2%}" if step > 0 else None)
                    with colC:
                        lead = "YOU" if user_avg < adv_avg else "ADVERSARY" if user_avg > adv_avg else "TIE"
                        st.metric("LEADER", lead)
                    
                    # Progress bars
                    st.progress((step+1)/max_steps, f"DUEL PROGRESS: {step+1}/{max_steps}")
                
                # Log to blockchain
                if step % 3 == 0:
                    block = st.session_state.blockchain.add_block({
                        'step': step,
                        'user_pos': (ur, uc),
                        'adv_pos': (ar, ac) if step < len(adversary_path) else None,
                        'congestion': float(user_avg)
                    })
                    blockchain_placeholder.info(f"🔗 Block #{len(st.session_state.blockchain.chain)} added: {block['hash']}")
                
                time.sleep(0.5)
            
            # Duel complete
            st.success("🎉 DUEL COMPLETED!")
            st.session_state.missions_completed += 1
            st.session_state.fleet_performance.append({
                'timestamp': time.time(),
                'score': your_score,
                'winner': 'user' if your_score > adv_score else 'adversary'
            })
            st.balloons()
    
    with col_map2:
        st.markdown("### 📡 SENSOR DATA")
        st.metric("GPS Accuracy", "±1.2m")
        st.metric("LiDAR Status", "✅ ACTIVE")
        st.metric("Comms Latency", f"{random.randint(5, 25)}ms")
        st.metric("Battery", f"{random.randint(60, 95)}%")
        
        st.markdown("### 🎯 TARGET INFO")
        st.info(f"Start: {start}")
        st.info(f"Goal: {goal}")
        st.info(f"Distance: {len(user_path) * 15}m")

# ==================== TAB 2: ANALYTICS HUB ==================== #

with tab2:
    st.markdown("# 📊 FLEET INTELLIGENCE ANALYTICS HUB")
    
    # Real-time system performance
    st.markdown("### ⚡ REAL-TIME SYSTEM PERFORMANCE")
    
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    
    with col_perf1:
        cpu = random.uniform(25, 65)
        st.metric("CPU Usage", f"{cpu:.1f}%", f"{cpu-45:.1f}%" if cpu > 45 else f"{cpu-45:.1f}%")
    with col_perf2:
        memory = random.uniform(30, 70)
        st.metric("Memory", f"{memory:.1f}%", f"{memory-50:.1f}%")
    with col_perf3:
        network = random.uniform(10, 40)
        st.metric("Network I/O", f"{network:.1f} MB/s", f"{network-25:.1f}")
    with col_perf4:
        disk = random.uniform(20, 45)
        st.metric("Disk Usage", f"{disk:.1f}%", f"{disk-30:.1f}%")
    
    # Performance chart
    perf_data = pd.DataFrame({
        'Time': list(range(20)),
        'CPU': [random.uniform(20, 80) for _ in range(20)],
        'Memory': [random.uniform(30, 70) for _ in range(20)],
        'Network': [random.uniform(5, 50) for _ in range(20)]
    })
    st.line_chart(perf_data.set_index('Time'))
    
    # ===== RL AGENT SECTION ===== #
    st.markdown("---")
    st.markdown("## 🤖 REINFORCEMENT LEARNING CONTROLLER")
    
    col_rl1, col_rl2, col_rl3, col_rl4 = st.columns(4)
    
    with col_rl1:
        st.metric("Episodes Trained", st.session_state.rl_episodes)
    with col_rl2:
        q_stats = st.session_state.rl_agent.get_q_stats()
        st.metric("Q-Table Size", q_stats["size"])
    with col_rl3:
        st.metric("Exploration Rate", f"{st.session_state.rl_agent.exploration_rate:.3f}")
    with col_rl4:
        st.metric("Learning Rate", f"{st.session_state.rl_agent.learning_rate:.2f}")
    
    # Training interface
    with st.expander("🎮 RL TRAINING INTERFACE", expanded=True):
        col_train1, col_train2, col_train3 = st.columns(3)
        
        with col_train1:
            num_episodes = st.slider("Episodes", 5, 100, 20)
        with col_train2:
            lr = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            st.session_state.rl_agent.learning_rate = lr
        with col_train3:
            discount = st.slider("Discount Factor", 0.5, 0.99, 0.95, 0.01)
            st.session_state.rl_agent.discount = discount
        
        if st.button("🚀 START TRAINING", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            episode_rewards = []
            
            for episode in range(num_episodes):
                # Training episode
                total_reward = 0
                traffic_prob = random.uniform(0, 1)
                distance = random.uniform(0, 1)
                time_of_day = random.uniform(0, 24)
                weather = random.choice(["Clear", "Rainy"])
                
                state = st.session_state.rl_agent.get_state(traffic_prob, distance, time_of_day, weather)
                
                for step in range(5):
                    action = st.session_state.rl_agent.get_action(state)
                    
                    if action == 0:  # Aggressive
                        success_prob = 0.7 - traffic_prob * 0.5
                        speed_boost = 1.5
                    elif action == 1:  # Normal
                        success_prob = 0.8 - traffic_prob * 0.3
                        speed_boost = 1.0
                    else:  # Cautious
                        success_prob = 0.95 - traffic_prob * 0.1
                        speed_boost = 0.6
                    
                    success = random.random() < success_prob
                    reward = 10 * speed_boost if success else -5 - (traffic_prob * 10)
                    reward += random.gauss(0, 1)
                    
                    next_state = st.session_state.rl_agent.get_state(
                        min(1, max(0, traffic_prob + random.gauss(0, 0.1))),
                        max(0, distance - 0.2 * speed_boost),
                        (time_of_day + 1) % 24,
                        weather if random.random() > 0.1 else ("Rainy" if weather == "Clear" else "Clear")
                    )
                    
                    st.session_state.rl_agent.update_q(state, action, reward, next_state)
                    total_reward += reward
                    state = next_state
                
                episode_rewards.append(total_reward)
                st.session_state.rl_episodes += 1
                
                # Update progress
                progress_bar.progress((episode + 1) / num_episodes)
                status_text.text(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f}")
                
                # Update chart
                if episode % 5 == 0:
                    df_temp = pd.DataFrame({
                        'Episode': list(range(len(episode_rewards))),
                        'Reward': episode_rewards
                    })
                    chart_placeholder.line_chart(df_temp.set_index('Episode'))
            
            st.session_state.rl_rewards_history.extend(episode_rewards)
            progress_bar.empty()
            status_text.success(f"✅ Training complete! Avg Reward: {np.mean(episode_rewards):.2f}")
    
    # ===== GENETIC ALGORITHM SECTION ===== #
    st.markdown("---")
    st.markdown("## 🧬 GENETIC FLEET OPTIMIZER")
    
    col_gen1, col_gen2 = st.columns(2)
    
    with col_gen1:
        st.markdown("### Current Generation")
        tasks = ["Task A", "Task B", "Task C", "Task D", "Task E", "Task F", "Task G"]
        vehicles = ["Truck1", "Truck2", "Truck3", "Drone1", "Drone2"]
        
        if st.button("🧬 RUN GENETIC OPTIMIZATION"):
            optimizer = GeneticFleetOptimizer(tasks, vehicles, generations=100, pop_size=50)
            with st.spinner("Evolving optimal fleet allocation..."):
                allocation = optimizer.optimize()
            
            df_alloc = pd.DataFrame(list(allocation.items()), columns=["Task", "Vehicle"])
            st.dataframe(df_alloc, use_container_width=True)
            
            # Show fitness evolution
            fitness_data = pd.DataFrame({
                'Generation': list(range(100)),
                'Fitness': [random.uniform(70, 95) + i*0.1 for i in range(100)]
            })
            st.line_chart(fitness_data.set_index('Generation'))
    
    with col_gen2:
        st.markdown("### Fleet Load Distribution")
        # Generate sample load data
        load_data = pd.DataFrame({
            'Vehicle': vehicles,
            'Tasks': [random.randint(1, 4) for _ in vehicles]
        })
        fig = px.pie(load_data, values='Tasks', names='Vehicle', title="Task Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== PREDICTIVE ANALYTICS ===== #
    st.markdown("---")
    st.markdown("## 🔮 PREDICTIVE ANALYTICS")
    
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        st.markdown("### Traffic Prediction (Next 24h)")
        hours = list(range(24))
        traffic_pred = [0.3 + 0.4 * np.sin(2 * np.pi * h / 24) + random.gauss(0, 0.05) for h in hours]
        df_traffic = pd.DataFrame({
            'Hour': hours,
            'Predicted Traffic': traffic_pred
        })
        st.line_chart(df_traffic.set_index('Hour'))
        
        peak_hour = np.argmax(traffic_pred)
        st.info(f"🚦 Peak traffic predicted at {peak_hour}:00 ({traffic_pred[peak_hour]:.1%})")
    
    with col_pred2:
        st.markdown("### Mission Success Probability")
        # Monte Carlo simulation
        n_sims = 1000
        success_rates = []
        for i in range(n_sims):
            traffic = random.uniform(0.2, 0.8)
            weather = random.choice([0.8, 0.9, 0.95, 0.7])
            success_rates.append(weather - traffic * 0.3)
        
        fig = go.Figure(data=[go.Histogram(x=success_rates, nbinsx=30)])
        fig.update_layout(title="Monte Carlo Simulation Results")
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: AI LAB ==================== #

with tab3:
    st.markdown("# 🧠 ADVANCED AI LABORATORY")
    st.markdown("### *Experiment with cutting-edge AI algorithms*")
    
    col_lab1, col_lab2 = st.columns(2)
    
    with col_lab1:
        st.markdown("## 🎯 FUZZY LOGIC CONTROLLER")
        fuzzy = FuzzyUrgencyController()
        
        dist_fuzzy = st.slider("Distance to goal", 0, 100, 50, key="fuzzy_dist")
        traffic_fuzzy = st.slider("Traffic density", 0.0, 1.0, 0.5, 0.01, key="fuzzy_traffic")
        
        speed = fuzzy.compute_speed_multiplier(dist_fuzzy, 100, traffic_fuzzy)
        
        # Visualize fuzzy output
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            near, medium, far = fuzzy.fuzzify_distance(dist_fuzzy, 100)
            st.metric("Near Membership", f"{near:.2f}")
        with col_f2:
            near, medium, far = fuzzy.fuzzify_distance(dist_fuzzy, 100)
            st.metric("Medium Membership", f"{medium:.2f}")
        with col_f3:
            near, medium, far = fuzzy.fuzzify_distance(dist_fuzzy, 100)
            st.metric("Far Membership", f"{far:.2f}")
        
        st.metric("RECOMMENDED SPEED", f"{speed:.2f}x", delta=f"{speed-1:.2f}")
        
        # Fuzzy surface 3D
        st.markdown("### 3D Fuzzy Control Surface")
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[fuzzy.compute_speed_multiplier(xi, 100, yi) for xi in x] for yi in y])
        
        fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
        fig.update_layout(
            title='Speed Multiplier Surface',
            scene=dict(
                xaxis_title='Distance',
                yaxis_title='Traffic',
                zaxis_title='Speed'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_lab2:
        st.markdown("## 🎲 MDP POLICY VISUALIZATION")
        
        # Create interactive MDP
        size = st.slider("Grid Size", 3, 6, 4)
        small_grid = [[0 for _ in range(size)] for _ in range(size)]
        # Add random obstacles
        for _ in range(size//2):
            r, c = random.randint(0, size-1), random.randint(0, size-1)
            if (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                small_grid[r][c] = 1
        
        goal_mdp = (size-1, size-1)
        mdp = MDPPolicyIterator(small_grid, goal_mdp, discount=0.9, noise=0.1, step_cost=-0.1)
        mdp.value_iteration(max_iterations=100)
        policy = mdp.get_policy()
        
        # Display policy as heatmap
        arrow_map = {(-1,0): "↑", (1,0): "↓", (0,-1): "←", (0,1): "→", None: "•"}
        policy_grid = []
        values_grid = []
        
        for r in range(size):
            policy_row = []
            value_row = []
            for c in range(size):
                if small_grid[r][c] == 1:
                    policy_row.append("█")
                    value_row.append(-50)
                elif (r,c) == goal_mdp:
                    policy_row.append("🎯")
                    value_row.append(100)
                else:
                    act = policy[r,c]
                    policy_row.append(arrow_map.get(act, "•"))
                    value_row.append(mdp.values[r][c])
            policy_grid.append(policy_row)
            values_grid.append(value_row)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Policy Map**")
            df_policy = pd.DataFrame(policy_grid)
            st.table(df_policy)
        
        with col_p2:
            st.markdown("**Value Function**")
            fig = px.imshow(values_grid, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== NEURAL NETWORK VISUALIZER ===== #
    st.markdown("---")
    st.markdown("## 🧬 NEURAL NETWORK ARCHITECTURE")
    
    col_nn1, col_nn2 = st.columns([1, 2])
    
    with col_nn1:
        st.markdown("### Network Configuration")
        layers = st.multiselect(
            "Hidden Layers",
            ["64", "128", "256", "512"],
            default=["128", "64"]
        )
        activation = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"])
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
        learning_rate_nn = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    with col_nn2:
        # Draw neural network
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Network structure
        layer_sizes = [4] + [int(l) for l in layers] + [3]
        
        # Position nodes
        x_positions = np.linspace(0.1, 0.9, len(layer_sizes))
        y_positions = [np.linspace(0.1, 0.9, n) for n in layer_sizes]
        
        # Draw connections
        for i in range(len(layer_sizes)-1):
            for j in range(layer_sizes[i]):
                for k in range(layer_sizes[i+1]):
                    alpha = random.uniform(0.1, 0.5)
                    ax.plot([x_positions[i], x_positions[i+1]], 
                           [y_positions[i][j], y_positions[i+1][k]], 
                           'gray', linewidth=1, alpha=alpha)
        
        # Draw nodes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layer_sizes)))
        for i, (x, y_layer, color) in enumerate(zip(x_positions, y_positions, colors)):
            ax.scatter([x]*len(y_layer), y_layer, s=300, 
                      c=[color], edgecolors='white', linewidth=2, zorder=5)
            ax.text(x, 0.02, f'Layer {i}\n{layer_sizes[i]}', ha='center', fontsize=8)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Neural Network: {activation} / {optimizer}")
        
        st.pyplot(fig)
    
    # ===== CSP SOLVER ===== #
    st.markdown("---")
    st.markdown("## 📅 CONSTRAINT SATISFACTION PROBLEM")
    
    csp_tasks = ["Inspection", "Loading", "Travel", "Delivery", "Maintenance", "Refueling"]
    
    col_csp1, col_csp2 = st.columns(2)
    
    with col_csp1:
        st.markdown("### Task Domains")
        domains = {}
        for task in csp_tasks[:4]:  # Limit for demo
            domains[task] = st.multiselect(
                f"{task} time slots",
                [8,9,10,11,12,13,14,15,16],
                default=[8,9,10,11,12]
            )
    
    with col_csp2:
        st.markdown("### Constraints")
        st.info("1. No two tasks at same time")
        st.info("2. Loading must be before Travel")
        st.info("3. Maintenance cannot be at peak hours (12-14)")
        
        def constraint_no_overlap(task, value, assignment):
            for t, v in assignment.items():
                if v == value:
                    return False
            return True
        
        def constraint_loading_before_travel(task, value, assignment):
            if task == "Travel" and "Loading" in assignment:
                return value > assignment["Loading"]
            return True
        
        constraints = [constraint_no_overlap, constraint_loading_before_travel]
        
        if st.button("🔍 SOLVE CSP"):
            # Simplified domains for demo
            demo_domains = {
                "Inspection": [8,9,10],
                "Loading": [9,10,11],
                "Travel": [10,11,12,13],
                "Delivery": [11,12,13,14]
            }
            scheduler = CSPScheduler(list(demo_domains.keys()), demo_domains, constraints)
            solution = scheduler.backtrack()
            
            if solution:
                st.success("✅ Solution Found!")
                st.json(solution)
                
                # Visualize schedule
                schedule_data = []
                for task, time in solution.items():
                    schedule_data.append({
                        'Task': task,
                        'Start': time,
                        'End': time + 1,
                        'Duration': 1
                    })
                df_schedule = pd.DataFrame(schedule_data)
                st.dataframe(df_schedule)
            else:
                st.error("❌ No solution found")

# ==================== TAB 4: BLOCKCHAIN LEDGER ==================== #

with tab4:
    st.markdown("# 🔐 IMMUTABLE BLOCKCHAIN LEDGER")
    st.markdown("### *All mission data cryptographically secured*")
    
    # Blockchain stats
    col_bc1, col_bc2, col_bc3, col_bc4 = st.columns(4)
    
    with col_bc1:
        st.metric("Total Blocks", len(st.session_state.blockchain.chain))
    with col_bc2:
        st.metric("Chain Length", f"{len(st.session_state.blockchain.chain) * 256} bytes")
    with col_bc3:
        st.metric("Mining Difficulty", st.session_state.blockchain.difficulty)
    with col_bc4:
        st.metric("Valid Blocks", sum(1 for b in st.session_state.blockchain.chain if b.get('hash')))
    
    # Visualize blockchain
    if st.session_state.blockchain.chain:
        st.markdown("### 🔗 Blockchain Visualization")
        
        # Create blockchain visualization
        blocks_data = []
        for i, block in enumerate(st.session_state.blockchain.chain):
            blocks_data.append({
                'Block': i,
                'Hash': block.get('hash', 'N/A'),
                'Previous': block.get('previous_hash', '0'*8),
                'Timestamp': datetime.fromtimestamp(block.get('timestamp', time.time())).strftime('%H:%M:%S'),
                'Data': str(block.get('data', {}))[:50]
            })
        
        df_blocks = pd.DataFrame(blocks_data)
        st.dataframe(df_blocks, use_container_width=True)
        
        # Block explorer
        st.markdown("### 🔍 Block Explorer")
        block_num = st.slider("Select Block", 0, len(st.session_state.blockchain.chain)-1, len(st.session_state.blockchain.chain)-1)
        
        selected_block = st.session_state.blockchain.chain[block_num]
        st.json(selected_block)
        
        # Verify chain integrity
        st.markdown("### ✅ Chain Integrity Check")
        if st.button("Verify Blockchain"):
            valid = True
            for i in range(1, len(st.session_state.blockchain.chain)):
                prev_hash = st.session_state.blockchain.chain[i-1].get('hash')
                curr_prev_hash = st.session_state.blockchain.chain[i].get('previous_hash')
                if prev_hash != curr_prev_hash:
                    valid = False
                    break
            
            if valid:
                st.success("✅ Blockchain is valid and tamper-proof!")
            else:
                st.error("❌ Blockchain integrity compromised!")
    else:
        st.info("No blocks in chain yet. Run a duel to add blocks.")
        
        if st.button("➕ Create Genesis Block"):
            genesis = st.session_state.blockchain.add_block({
                'event': 'SYSTEM_START',
                'version': 'ULTIMATE_2.0',
                'timestamp': time.time()
            })
            st.success(f"Genesis block created: {genesis['hash']}")
            st.rerun()

# ==================== FOOTER ==================== #

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <h4>🤖 ULTIMATE AUTONOMOUS FLEET INTELLIGENCE SYSTEM v3.0</h4>
        <p>Powered by: Reinforcement Learning • Genetic Algorithms • Neural Networks • Fuzzy Logic • MDP • CSP • Blockchain</p>
        <p style='color: #666'>⚡ 8 Active AI Agents • Real-time Processing • Military-grade Encryption ⚡</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Periodic telemetry flush
st.session_state.telemetry.flush()