[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/f5L5wp83)

# Drone MDP Challenge

## Overview

In this project, you will model a drone navigation problem as a **Markov Decision Process (MDP)** and compute an optimal policy using the **Value Iteration** algorithm.

The drone operates in a stochastic grid-world environment containing hazards, portals, repair stations, and a goal state. Your objective is to maximize the expected cumulative reward while safely reaching the target.

---

## Learning Objectives

By completing this project, you will:

- Model a real-world problem as an MDP
- Implement the Value Iteration algorithm
- Work with stochastic transitions
- Design and analyze reward functions
- Extract optimal policies from value functions
- Visualize value functions and policies

---

## Environment

The environment is a randomly generated grid with dimensions ranging from **9×9** to **15×15**.

Each state is represented as:

```python
(row, column, damage)
```

where:

- `row` : grid row
- `column` : grid column
- `damage` : drone damage level (0–4)

A damage level of **5** represents a crash and terminates the episode.

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

## Task

Implement the policy computation function using **Value Iteration**.

Your implementation should:

1. Read environment parameters
2. Retrieve all states
3. Initialize state values
4. Perform Value Iteration until convergence
5. Extract the optimal action for each state
6. Return the final policy

Expected output format:

```python
{
    (row, col, damage): action
}
```

---

## Required Visualizations

In addition to the policy implementation, submit a visualization script that generates the following figures.

### 1. Value Function Heatmap

Visualize state values on the grid.

- One heatmap for each damage level

### 2. Convergence Curve

Plot the maximum value update per iteration.

The curve should demonstrate convergence of the Value Iteration process.

### 3. Policy Visualization

Visualize the final policy using directional arrows.

- One figure for each damage level
- Goal and wall cells should be clearly distinguished

---

## Evaluation

Your policy will be evaluated over **5 independent episodes**.

| Criterion | Weight |
|------------|----------|
| Goal Reach Rate | 40% |
| Total Reward | 30% |
| Crash Avoidance | 15% |
| Policy Coverage | 15% |

Additional credit may be awarded for accurately modeling and utilizing all environment features.

---

## Running the Project

Install dependencies:

```bash
pip install flask flask-cors
```

Run the server:

```bash
python server.py
```

Then:

1. Implement the policy function
2. Launch the provided interface
3. Observe the execution of five evaluation episodes

---

## Deliverables

### Required Files

- `policy.py`
- Visualization script
- Technical report (PDF)

### Technical Report

Your report should include:

- MDP formulation
- Value Iteration implementation details
- Reward modeling
- Transition modeling
- Policy extraction process
- Experimental results
- Required visualizations

---

## Course Information

**Course:** Fundamentals and Applications of Artificial Intelligence  
**Project:** Drone MDP Challenge  
**Instructor:** Dr. Marzieh Hosseini  
**Teaching Assistants:** Masih Roughani, Marzieh Karami, Fatemeh Sayadzade  
**Department:** Faculty of Computer Engineering, University of Isfahan  
**Semester:** Spring 2026
