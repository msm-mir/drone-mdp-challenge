class MDP_class:
    def __init__(self, api):
        self.server_api = api
        
        self.medkits = []
        self.init_medkits()

        self.medkit_offset = -15 # -15
        self.medkit_mult = 20 # 10

        self.storm_offset = -50 # -50
        self.storm_mult = -30 # -30

        self.stormZone_offset = -20 # -20
        self.stormZone_mult = 0 # 0

        self.wall_offset = -10 # -10
        self.wall_mult = 0 # 0

    def init_medkits(self):
        for r in range(self.server_api._rows):
            for c in range(self.server_api._cols):
                if self.server_api._cell(r, c) == 'medkit':
                    self.medkits.append((r, c))

    def get_all_states(self):
        # get states from server
        base_states = self.server_api.get_all_states() # (r, c, dmg)
        states = []

        for r, c, dmg in base_states:
            # 2^medkit_count, 0 means medkit exists
            for mask in range(1 << len(self.medkits)):
                states.append((r, c, dmg, mask))
        return states
    
    def get_transitions(self, state, action):
        r, c, dmg, mask = state
        base = self.server_api.get_transitions((r, c, dmg), action)
        result = []

        for (nr, nc, ndmg), prob in base:
            new_mask = mask
            # If this move lands on a medkit
            if (nr, nc) in self.medkits:
                idx = self.medkits.index((nr, nc))
                if not (mask >> idx) & 1:
                    new_mask = mask | (1 << idx) # mark medkit as collected
            result.append(((nr, nc, ndmg, new_mask), prob))
        return result
    
    def get_reward(self, state, action, next_state):
        r, c, dmg, mask = state
        nr, nc, ndmg, nmask = next_state

        reward = { 'base': 0, 'total': 0, 'goal': 0, 'medkit': 0, 'portal': 0, 
                  'obstacle': 0, 'storm': 0, 'storm_zone': 0 }

        # get base reward from server
        reward['base'] = self.api.get_reward((r, c, dmg), action, (nr, nc, ndmg))
        reward['total'] = reward['base']

        ### ==========ADJUST THE REWARD==========
        t = self.api._cell(nr, nc)
        tmp_reward = 0

        # the next cell is a medkit
        if (nr, nc) in self.medkits:
            idx = self.medkits.index((nr, nc))
            was_collected = (mask >> idx) & 1

            if was_collected:
                tmp_reward = -25
            else:
                tmp_reward = self.medkit_offset + dmg * self.medkit_mult
            reward['total'] += tmp_reward
            
            # base reward of medkit + our reward of medkit
            reward['medkit'] = 25 + tmp_reward - self.api.DMG_COST[dmg]
        
        # the next cell is storm
        if t == "storm":
            tmp_reward = self.storm_offset + dmg * self.storm_mult
            reward['total'] += tmp_reward

            # base reward of storm + our reward of storm
            sev = self.api._storm_sev.get(f'{nr},{nc}', 1)
            reward['storm'] = self.api.STORM_REWARD[sev] + tmp_reward - self.api.DMG_COST[dmg]

        # the next cell is in storm zone
        if self.api._in_zone(nr, nc):
            z = self.api._storm_zone
            if z:
                tmp_reward = self.stormZone_offset + dmg * self.stormZone_mult
                reward['total'] += tmp_reward

                # base reward of storm zone + our reward of storm zone
                reward['storm_zone'] = -z['eExpected'] + tmp_reward - self.api.DMG_COST[dmg]
        
        # the next cell is wall
        if r == nr and c == nc:
            tmp_reward = self.wall_offset + dmg * self.wall_mult
            reward['total'] += tmp_reward

            # base reward of wall + our reward of wall
            reward['obstacle'] = -self.api.WALL_PENALTY + tmp_reward - self.api.DMG_COST[dmg]
        
        # the next cell is portal
        if t in ('portal_a', 'portal_b'):
            tmp_reward = 0
            # base reward of portal + our reward of portal
            reward['portal'] = -5 + tmp_reward - self.api.DMG_COST[dmg]

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