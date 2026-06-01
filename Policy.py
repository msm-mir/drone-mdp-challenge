"""
policy.py — Student Code
─────────────────────────────────────────────────────────────────────────────
The only function you need to implement: compute_policy(api)

Input:  api   — A DroneAPI object provided by server.py:
    api.get_all_states()                → list[tuple(row,col,damage)]
    api.get_possible_actions(state)     → list[str]  (empty if terminal)
    api.get_transitions(state, action)  → list[(next_state, prob)]
    api.get_reward(state, action, next_state) → float
    api.is_terminal(state)             → bool
    api.get_env_params()               → dict  {'fwd','left','right','gamma','rows','cols','portals'}
    api.get_storm_zone()               → dict | None  {r1,c1,r2,c2,eMin,eMax,eExpected}

Output: dict  — mapping state → action
    Key:   tuple (row, col, damage)  OR  string "row,col,damage"
    Value: one of 'N', 'S', 'E', 'W'

Scoring (per episode, max 5 episodes):
    Goal reached  : +300
    Medkit (fresh): +25, damage -1
    Storm S1      : -50,  damage +1
    Storm S2      : -100, damage +1
    Storm S3      : -125, damage +2
    Portal use    : -5
    Wall hit      : -3 extra
    Storm zone    : -random(eMin..eMax) each step inside
    Step cost     : -DMG_COST[damage]  where DMG_COST = [1,2,5,9,15]
    Timeout (80)  : -250
    Crash (dmg=5) : -200
─────────────────────────────────────────────────────────────────────────────
"""


def compute_policy(api):
    params = api.get_env_params()
    gamma = params['gamma']

    THETA = 1e-4
    MAX_ITER = 500

    states = api.get_all_states()
    V = {s: 0.0 for s in states}
    policy = {}

    for iteration in range(MAX_ITER):
        delta = 0.0

        for s in states:
            if api.is_terminal(s):
                V[s] = 0.0
                continue

            actions = api.get_possible_actions(s)
            if not actions:
                continue

            best_val = float('-inf')
            for a in actions:
                q = sum(
                    prob * (api.get_reward(s, a, ns) + gamma * V.get(ns, 0.0))
                    for ns, prob in api.get_transitions(s, a)
                )
                if q > best_val:
                    best_val = q

            delta = max(delta, abs(best_val - V[s]))
            V[s] = best_val

        if delta < THETA:
            print(f"[policy.py] Converged in {iteration + 1} iterations (delta={delta:.2e})")
            break
    else:
        print(f"[policy.py] Reached MAX_ITER={MAX_ITER} (delta={delta:.2e})")

    for s in states:
        if api.is_terminal(s):
            continue

        actions = api.get_possible_actions(s)
        if not actions:
            continue

        best_action, best_q = None, float('-inf')
        for a in actions:
            q = sum(
                prob * (api.get_reward(s, a, ns) + gamma * V.get(ns, 0.0))
                for ns, prob in api.get_transitions(s, a)
            )
            if q > best_q:
                best_q, best_action = q, a

        if best_action:
            policy[s] = best_action

    print(f"[policy.py] Policy covers {len(policy)} states")
    return policy
