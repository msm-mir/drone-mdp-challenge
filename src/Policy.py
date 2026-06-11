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
        
        if t == 'goal':
            tmp_reward = 300
            # base reward of goal + our reward of goal
            reward['goal'] = tmp_reward

        return reward
    
    def is_terminal(self, state):
        r, c, dmg, mask = state
        return self.api.is_terminal((r, c, dmg)) or dmg == 5
    
    def get_possible_actions(self, state):
        r, c, dmg, mask = state
        return self.api.get_possible_actions((r, c, dmg))
    
    def get_env_params(self):
        return self.api.get_env_params()
    
    def _is_obstacle(self, r, c):
        return self.api._is_obstacle(r, c)

def compute_policy(api):
    my_api = MDP_class(api)
    params = api.get_env_params()
    gamma = params['gamma']

    THETA = 1e-4
    MAX_ITER = 500
    for_plotting = False

    states = my_api.get_all_states()

    delta_history = { 'base': [], 'total': [], 'goal': [], 'medkit': [], 'portal': [], #11
                      'obstacle': [], 'storm': [], 'storm_zone': [] }
    value_history = { 'base': {}, 'total': {}, 'goal': {}, 'medkit': {}, 'portal': {}, 
                      'obstacle': {}, 'storm': {}, 'storm_zone': {} }
    
    # initialization value_history
    for k in value_history.keys():
        value_history[k] = {s: 0.0 for s in states}
    
    # value iteration
    for type in value_history.keys():
        if (not for_plotting) and (type != 'total'): continue

        for iteration in range(MAX_ITER):
            delta = 0.0
            V_new = value_history[type].copy()

            for s in states:
                if my_api.is_terminal(s):
                    V_new[s] = 0.0
                    continue

                actions = my_api.get_possible_actions(s)
                if not actions:
                    continue

                best_val = float('-inf')
                for a in actions:
                    q = 0.0
                    for ns, prob in my_api.get_transitions(s, a):
                        reward = my_api.get_reward(s, a, ns)

                        q += prob * (reward[type] + gamma * value_history[type].get(ns, 0.0))

                    best_val = max(best_val, q)

                delta = max(delta, abs(best_val - V_new[s]))
                V_new[s] = best_val

            value_history[type] = V_new
            delta_history[type].append(delta)
            
            if delta < THETA:
                print(f"[policy.py] Converged in {iteration + 1} iterations (delta={delta:.2e})")
                break
        else:
            print(f"[policy.py] Reached MAX_ITER={MAX_ITER} (delta={delta:.2e})")

    policy = { 'base': {}, 'total': {}, 'goal': {}, 'medkit': {}, 'portal': {}, 
                      'obstacle': {}, 'storm': {}, 'storm_zone': {} }
    
    # compute policy
    for type in value_history.keys():
        if (not for_plotting) and (type != 'total'): continue
        
        for s in states:
            r, c, dmg, mask = s
            if mask != 0:
                continue

            if my_api.is_terminal(s):
                continue

            actions = my_api.get_possible_actions(s)
            if not actions:
                continue

            best_action, best_q = None, float('-inf')
            for a in actions:
                q = 0
                for ns, prob in my_api.get_transitions(s, a):
                    reward = my_api.get_reward(s, a, ns)

                    q += prob * (reward[type] + gamma * value_history[type].get(ns, 0.0))
                
                if q > best_q:
                    best_q, best_action = q, a

            if best_action:
                policy[type][(r, c, dmg)] = best_action
                value_history[type][(r, c, dmg)] = best_q

    print(f"[policy.py] Policy covers {len(policy)} states")
    return policy['total']