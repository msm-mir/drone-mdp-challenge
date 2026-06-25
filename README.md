# Drone MDP Challenge

## Overview

This project models a drone navigation problem as a **Markov Decision Process (MDP)** and computes an optimal policy using the **Value Iteration** algorithm.

The drone operates in a stochastic grid-world environment containing hazards, portals, repair stations, and a goal state. The objective is to maximize the expected cumulative reward while safely reaching the target.

---

## Objectives

- Model a real-world problem as an MDP
- Implement the Value Iteration algorithm
- Work with stochastic transitions
- Design and analyze reward functions
- Extract optimal policies from value functions
- Visualize value functions and policies

---

## Environment

The environment is a randomly generated grid with dimensions ranging from **9×9** to **15×15**.
- `row` : grid row
- `column` : grid column
- `damage` : drone damage level (0–4)
- `storm zone` : the penalty inside this zone is random, but the expected value (average) is used in the calculations.
- `medkit` : these items are single-use and disappear permanently.
- `damage impact` : the damage level determines the cost of each normal step.
- `crash condition` : a damage level of **5** represents a crash and terminates the episode.

---

## Cell Types

| Cell Type | Effect |
|------------|---------|
| Normal Cell | Standard movement cost |
| Wall | Cannot be entered |
| Goal | Terminal state with +300 reward |
| Storm Level 1 | -50 reward, damage +1 |
| Storm Level 2 | -100 reward, damage +1 |
| Storm Level 3 | -125 reward, damage +2 |
| First Aid | +25 reward, damage -1 (single use) |
| Portal | Teleports to paired portal, -5 reward |
| Storm Region | Additional expected penalty |

---

## Actions

The drone can choose one of the following actions:

- North
- South
- East
- West

Actions are **stochastic**. The drone may deviate from the intended direction according to the transition probabilities provided by the environment.

---

## Visualizations

### 1. Value Function Heatmap

Visualize state values on the grid.

- One heatmap for each damage level

### 2. Convergence Curve

Plot the maximum value update per iteration.

The curve demonstrates convergence of the Value Iteration process.

### 3. Policy Visualization

Visualize the final policy using directional arrows.

- One figure for each damage level
- Goal and wall cells are clearly distinguished

---

## Requirements

- Python 3.x
- flask
- flask_cors
- waitress

---

## Project Structure

```
├── app/
│   └── drone_mdp_v2.html
├── src/
│   ├── Policy.py
│   ├── Server.py
│   └── Visualize.py
├── visualizations/
│   ├── convergence_total.png
│   ├── value_heatmap_total.png
│   └── policy_map_total.png
├── .gitignore
└── README.md
```

---

## Course Information

**Course:** Fundamentals and Applications of Artificial Intelligence  
**Project:** Drone MDP Challenge  
**Instructor:** Dr. Marzieh Hosseini  
**Teaching Assistants:** Masih Roughani, Marzieh Karami, Fatemeh Sayadzade  
**Department:** Faculty of Computer Engineering, University of Isfahan  
**Semester:** Spring 2026
