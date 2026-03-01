# 🤖 Ultimate Fleet AI System  
### Multi-Algorithm Duel Simulation with Real-Time 3D Visualization & Statistical Validation

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

---

## 🌟 Overview

**Ultimate Fleet AI System** is a research-oriented multi-algorithm simulation framework that evaluates and compares diverse AI paradigms under a unified objective function.

The system integrates:

- Real-time 3D simulation
- 500–1000 run benchmark pipelines
- Statistical validation (t-tests, 95% confidence intervals)
- Blockchain-secured telemetry logging
- Modular AI experimentation lab

Unlike demo-style visualizers, this platform ensures **fair, statistically valid algorithm comparison** through a shared optimization objective.

---

## 🧠 Core Algorithms Implemented

| Algorithm | Category | Time Complexity | Optimality |
|------------|----------|----------------|------------|
| A* | Graph Search | O(b^d) | Optimal (admissible heuristic) |
| Q-Learning | Reinforcement Learning | O(|S||A|·episodes) | Asymptotically Optimal |
| Genetic Algorithm | Evolutionary | O(pop·gen·fitness) | Probabilistic |
| Value Iteration | Dynamic Programming | O(|S|²|A|·iter) | Optimal |
| Fuzzy Logic | Rule-Based | O(n_rules) | Interpretable |
| Hidden Markov Model | Probabilistic | O(T·N²) | ML Estimation |
| Bayesian Network | Probabilistic | O(2^n) | Exact Inference |
| CSP (Backtracking) | Constraint Solving | O(d^n) | Complete |

---

## 🎯 Unified Cost Function (The Equalizer)

All algorithms optimize the same objective:

```
J = α·travel_time + β·traffic + γ·weather_risk + δ·fuel
```

This ensures:

- Fair comparison  
- Reproducibility  
- Statistical validity  
- Consistent benchmarking  

---

## 🏗 System Architecture

### 1️⃣ Frontend Layer
- Streamlit UI (5 interactive tabs)
- Real-time controls
- Dynamic cost weight adjustment

### 2️⃣ Core Simulation Engine
- Shared cost function
- Modular algorithm registry
- Scenario generation engine
- Traffic and weather modeling

### 3️⃣ Statistical Engine
- Multi-simulation benchmarking
- Independent sample t-tests
- 95% confidence intervals
- Comparative matrix generation

### 4️⃣ Visualization Layer
- PyDeck 3D rendering
- Plotly statistical charts
- Matplotlib analytics

### 5️⃣ Telemetry & Blockchain Layer
- SQLite mission database
- SHA-256 hash chaining
- Immutable block verification

---

## 🔄 Data Flow

User Input  
→ Streamlit UI  
→ Simulation Engine  
→ Algorithm Execution  
→ Result Aggregation  
→ Statistical Analysis  
→ 3D Rendering & Charts  
→ Telemetry Logging  
→ Blockchain Hash Linking  

---

## 📊 Benchmark Pipeline

- 5 algorithms per benchmark
- 500–1000 simulations each
- 5 traffic conditions
- Mean & Standard Deviation computation
- Independent T-Test
- 95% Confidence Interval
- Final Comparison Matrix

---

## 🔬 Statistical Validation

The system prevents false conclusions by validating:

- Null hypothesis testing
- Statistical significance
- Variance comparison
- Confidence interval overlap

This ensures performance differences are not due to randomness.

---

## 🔐 Blockchain Logging System

Each mission step is logged as a chained block:

```sql
CREATE TABLE mission_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    agent_id TEXT,
    current_x INTEGER,
    current_y INTEGER,
    traffic_prob REAL,
    system_status TEXT,
    metadata TEXT,
    block_hash TEXT,
    previous_hash TEXT
);

CREATE INDEX idx_timestamp ON mission_logs(timestamp);
CREATE INDEX idx_agent ON mission_logs(agent_id);
CREATE INDEX idx_block_hash ON mission_logs(block_hash);
```

Each new entry stores:
- SHA-256 hash
- Previous block reference
- Tamper detection capability

---

## 🧩 Project Structure

```
fleet-ai-system/
│
├── main.py                  # Streamlit orchestrator
├── algorithms/
│   ├── astar.py
│   ├── qlearning.py
│   ├── genetic.py
│   ├── mdp.py
│   ├── fuzzy.py
│   ├── hmm.py
│   ├── bayesian.py
│   └── csp.py
│
├── database/
│   ├── telemetry.py
│   └── blockchain.py
│
├── visualization/
│   ├── duel_arena.py
│   ├── benchmark_dashboard.py
│   └── charts.py
│
├── logs.db
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/fleet-ai-system.git
cd fleet-ai-system
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run main.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧪 Research Usage

1. Adjust cost weights (α, β, γ, δ)
2. Select traffic condition
3. Run benchmark (500+ simulations)
4. Export results as CSV
5. Perform additional statistical analysis

---

## 🌍 Deployment

### Local
- Python 3.10+
- Streamlit Server
- SQLite database

### Cloud (Optional)
- Push to GitHub
- Deploy on Streamlit Cloud
- Replace SQLite with PostgreSQL for scale
- Add authentication layer

---

## 📈 Performance Characteristics

- Modular plug-and-play algorithm integration
- Real-time rendering at sub-second refresh
- Persistent mission logging
- Deterministic benchmarking
- Extensible architecture

---

## 🎯 Why This Project Matters

This system demonstrates:

- Strong algorithmic foundations
- Reinforcement learning implementation
- Statistical rigor
- Scalable modular design
- Applied blockchain mechanics
- Production-grade software structure
- Research reproducibility mindset

It bridges:

Theory ↔ Simulation ↔ Visualization ↔ Statistical Proof

---

## 👨‍💻 Author

**Vishnu Aravind**  
Integrated M.Sc. Theoretical Computer Science  
PSG College of Technology  

Interests:
- Reinforcement Learning
- Algorithm Design
- Simulation Systems
- Applied Game Theory
- AI Research Engineering

---

## 📜 License

MIT License

---

## 🤝 Contributions

Pull requests are welcome.  
For major changes, please open an issue first to discuss improvements.

---

## ⭐ Support

If you find this project useful, consider starring the repository.

---