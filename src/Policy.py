class MDP_class:
    def __init__(self, api):
        self.server_api = api
        
        self.medkits = []
        self.init_medkits()

    def init_medkits(self):
        for r in range(self.server_api._rows):
            for c in range(self.server_api._cols):
                if self.server_api._cell(r, c) == 'medkit':
                    self.medkits.append((r, c))

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
        V_new = V.copy()

        for s in states:
            if api.is_terminal(s):
                V_new[s] = 0.0
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
            V_new[s] = best_val

        V = V_new.copy()
        
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