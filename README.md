# Comparative Benchmarking of Reinforcement Learning vs Classical Planning  
## Stochastic Route Optimization under Uncertainty

A research-oriented benchmarking framework comparing **Reinforcement Learning (Q-Learning)** against classical planning algorithms (**A\***, **Value Iteration (MDP)**, **Genetic Algorithm**, and **Random baseline**) for stochastic fleet routing.

The system provides reproducible experiments, statistical validation, regret analysis, and interactive visualization via a Streamlit dashboard.

---

## 1. Problem Formulation

We model stochastic route optimization as a **Markov Decision Process (MDP)**:

**State**  
S = (x, y, traffic, weather, fuel)

**Action**  
A = {cautious, normal, aggressive}

**Reward**  
R(s,a) = - [ α·travel_time + β·traffic + γ·weather + δ·fuel ]

**Terminal Condition**
- Goal reached  
- Timeout  

All algorithms optimize the same unified cost function to ensure fair comparison.

---

## 2. Algorithms Compared

| Algorithm | Category | Optimality | Notes |
|------------|----------|------------|--------|
| Q-Learning | Reinforcement Learning | Asymptotically optimal | Model-free, trained on stochastic environment |
| A* | Graph Search | Optimal (admissible heuristic) | Deterministic baseline |
| Value Iteration | Dynamic Programming | Optimal | Requires known transition model |
| Genetic Algorithm | Evolutionary | Probabilistic | Population-based search |
| Random | Baseline | None | Lower bound |

---

## 3. Experimental Design

- 100–1000 Monte Carlo simulations per algorithm per condition  
- 5 traffic conditions:
  - Low  
  - Medium  
  - High  
  - Stochastic (HMM-driven)  
  - Adversarial  
- Fixed random seeds for reproducibility  
- 95% confidence intervals  
- Independent two-sample t-tests  
- Cumulative regret relative to MDP  

**Training Protocol**
- RL is trained once on a base stochastic environment  
- Evaluation is performed with greedy policy (no exploration)  
- Optional grid randomization tests generalization  

---

## 4. System Architecture

```
Streamlit Dashboard
│
├── Benchmark Engine
│   ├── RL (Q-Learning)
│   ├── A*
│   ├── MDP (Value Iteration)
│   ├── Genetic Algorithm
│   └── Random Baseline
│
├── Stochastic Environment
│   ├── HMM Traffic Model
│   ├── Bayesian Weather Model
│   └── Unified Cost Function
│
└── Blockchain Logger (SHA-256)
```

## 5. Dashboard Features

### Duel Arena (3D)
Real-time fleet visualization using PyDeck with congestion tracking.

### Benchmark & Statistics
- Cost distribution plots  
- Mean ± standard deviation  
- 95% confidence intervals  
- Statistical significance testing  

### RL Analysis
- Learning curves (steps per episode)  
- Moving average smoothing  
- Cumulative regret vs MDP  

### AI Laboratory
- Fuzzy logic controller  
- MDP policy visualization  
- Neural network activation visualizer  

### Blockchain Ledger
- SHA-256 mission logging  
- Hash chaining  
- Immutable audit display  

---

## 6. Key Insights (Typical Observations)

- RL converges to near-optimal performance after ~500 episodes.  
- MDP shows lowest variance due to model-based planning.  
- A* performs well in static conditions but degrades under stochasticity.  
- Genetic algorithm shows higher variance but can escape local minima.  
- Random baseline establishes lower bound performance.  

Actual results vary depending on seed and grid configuration.

---

## 7. Technology Stack

- Python 3.10+
- Streamlit
- NumPy
- Pandas
- SciPy
- Plotly
- Matplotlib
- PyDeck
- SQLite
- SHA-256 (hashlib)

All algorithms are implemented from scratch. No external RL libraries are used.

---

## 8. Installation

Clone the repository:

```
git clone https://github.com/VishnuAravind-RG/dynamic-duel-simulation-with-adversary-tracking-and-live-dashboard.git
cd dynamic-duel-simulation-with-adversary-tracking-and-live-dashboard
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run main.py
```

Then open:

http://localhost:8501
---

## 9. Reproducibility

- Random seed control available in sidebar  
- Fixed-grid and randomized-grid evaluation modes  
- Deterministic training when seed is fixed  
- Monte Carlo simulation count configurable  

---

## 10. Limitations

- Q-table does not scale to very large state spaces  
- RL state abstraction may simplify full MDP definition  
- Genetic algorithm uses simplified chromosome encoding  
- Environment is synthetic (not real-world traffic API)  

This project focuses on algorithmic comparison, not production deployment.

---

## 11. Future Work

- Deep Q-Network (DQN)  
- Policy gradient methods (PPO, A2C)  
- Multi-agent coordination  
- Real-world traffic API integration  
- Cloud deployment (Docker/Kubernetes)  

---

## 12. Author

**Vishnu Aravind**  
Integrated M.Sc. Theoretical Computer Science  
PSG College of Technology  

Research Interests:
- Reinforcement Learning  
- Multi-Agent Systems  
- Algorithmic Decision Theory  
- AI for Logistics  

---

## License

MIT License