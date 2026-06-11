from flask import Flask, request, jsonify
from flask_cors import CORS
import importlib, importlib.util, sys, os, traceback, math, random
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

ENV = {}
PARAMS = {}


class DroneAPI:
    DMG_COST = [1, 2, 5, 9, 15]
    WALL_PENALTY = 3
    GOAL_REWARD = 300
    STORM_REWARD = {1: -50, 2: -100, 3: -125}
    STORM_DMG_INC = {1: 1, 2: 1, 3: 2}
    MAX_DAMAGE = 4
    MOVE = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
    PERP = {'N': ['W', 'E'], 'S': ['E', 'W'],
            'E': ['N', 'S'], 'W': ['S', 'N']}

    def __init__(self, env, params):
        self._env = env
        self._params = params
        self._grid = env['grid']
        self._rows = env['rows']
        self._cols = env['cols']
        self._storm_sev = env.get('stormSev', {})
        self._storm_zone = env.get('stormZone')
        self._portals = env.get('portalPairs', [])

    def _cell(self, r, c):
        return self._grid[r][c]

    def _in_bounds(self, r, c):
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _is_obstacle(self, r, c):
        return self._cell(r, c) == 'obstacle'

    def _portal_dest(self, r, c):
        for pair in self._portals:
            a, b = pair['a'], pair['b']
            if a['row'] == r and a['col'] == c:
                return (b['row'], b['col'])
            if b['row'] == r and b['col'] == c:
                return (a['row'], a['col'])
        return None

    def _in_zone(self, r, c):
        z = self._storm_zone
        if not z:
            return False
        return z['r1'] <= r <= z['r2'] and z['c1'] <= c <= z['c2']

    def _try_move(self, row, col, action):
        dr, dc = self.MOVE[action]
        nr, nc = row + dr, col + dc
        if not self._in_bounds(nr, nc) or self._is_obstacle(nr, nc):
            return (row, col, True)
        return (nr, nc, False)

    def _resolve_portal(self, r, c):
        t = self._cell(r, c)
        if t in ('portal_a', 'portal_b'):
            dest = self._portal_dest(r, c)
            if dest:
                return dest[0], dest[1], True
        return r, c, False

    def _next_damage(self, dmg, dest_r, dest_c):
        t = self._cell(dest_r, dest_c)
        if t == 'storm':
            sev = self._storm_sev.get(f'{dest_r},{dest_c}', 1)
            return dmg + self.STORM_DMG_INC[sev]
        if t == 'medkit':
            return max(dmg - 1, 0)
        return dmg

    def get_env_params(self):
        return {
            'fwd': self._params['fwd'],
            'left': self._params['left'],
            'right': self._params['right'],
            'gamma': self._params['gamma'],
            'rows': self._rows,
            'cols': self._cols,
            'portals': self._portals,
        }

    def get_storm_zone(self):
        return self._storm_zone

    def get_all_states(self):
        states = []
        for r in range(self._rows):
            for c in range(self._cols):
                t = self._cell(r, c)
                if t == 'obstacle':
                    continue
                for d in range(self.MAX_DAMAGE + 1):
                    states.append((r, c, d))
        return states

    def get_possible_actions(self, state):
        r, c, dmg = state
        if self._cell(r, c) == 'goal':
            return []
        return ['N', 'S', 'E', 'W']

    def get_transitions(self, state, action):
        row, col, dmg = state
        fwd = self._params['fwd']
        left = self._params['left']
        right = self._params['right']
        perps = self.PERP[action]

        outcomes = {
            action: fwd,
            perps[0]: left,
            perps[1]: right,
        }

        result = {}
        for act, prob in outcomes.items():
            if prob == 0:
                continue
            nr, nc, wall = self._try_move(row, col, act)
            pr, pc, was_portal = self._resolve_portal(nr, nc)
            new_dmg = min(self._next_damage(dmg, pr, pc), self.MAX_DAMAGE)
            ns = (pr, pc, new_dmg)
            result[ns] = result.get(ns, 0) + prob

        return list(result.items())

    def get_reward(self, state, action, next_state):
        r, c, dmg = state
        nr, nc, ndmg = next_state

        t = self._cell(nr, nc)

        if t == 'goal':
            return self.GOAL_REWARD

        wall_hit = (r == nr and c == nc)

        reward = 0.0

        if t == 'storm':
            sev = self._storm_sev.get(f'{nr},{nc}', 1)
            reward += self.STORM_REWARD[sev]
        elif t == 'medkit':
            reward += 25
        elif t in ('portal_a', 'portal_b'):
            reward += -5
        else:
            reward += -self.DMG_COST[dmg]

        if wall_hit:
            reward -= self.WALL_PENALTY

        if t != 'goal' and self._in_zone(nr, nc):
            z = self._storm_zone
            if z:
                reward -= z['eExpected']

        return reward

    def is_terminal(self, state):
        r, c, _ = state
        return self._cell(r, c) == 'goal'

    def get_cell_type(self, state):
        r, c, _ = state
        return self._cell(r, c)

    def get_storm_severity(self, state):
        r, c, _ = state
        return self._storm_sev.get(f'{r},{c}', 1)

    def is_in_storm_zone(self, state):
        r, c, _ = state
        return self._in_zone(r, c)


@app.route('/api/ping')
def ping():
    return jsonify({'ok': True, 'version': 'drone-mdp-v6'})


@app.route('/api/init', methods=['POST'])
def init():
    global ENV, PARAMS
    data = request.get_json()
    ENV = {
        'rows': data['rows'],
        'cols': data['cols'],
        'grid': data['grid'],
        'stormSev': data.get('stormSev', {}),
        'stormZone': data.get('stormZone'),
        'portalPairs': data.get('portalPairs', []),
        'startPos': data.get('startPos', {'row': 0, 'col': 0}),
    }
    PARAMS = {
        'fwd': data['fwd'],
        'left': data['left'],
        'right': data['right'],
        'gamma': data['gamma'],
    }
    return jsonify({'ok': True})


@app.route('/api/compute_policy', methods=['POST'])
def compute_policy():
    if not ENV:
        return jsonify({'ok': False, 'error': 'Map not initialised — call /api/init first'})

    policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policy.py')
    if not os.path.exists(policy_path):
        return jsonify({'ok': False, 'error': f'policy.py not found at: {policy_path}'})

    try:
        spec = importlib.util.spec_from_file_location('policy', policy_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, 'compute_policy'):
            return jsonify({'ok': False, 'error': 'policy.py must define compute_policy(api)'})

        api = DroneAPI(ENV, PARAMS)
        result = mod.compute_policy(api)

        if not isinstance(result, dict):
            return jsonify({'ok': False, 'error': f'compute_policy must return a dict, got {type(result)}'})

        policy = {}
        for k, v in result.items():
            if isinstance(k, tuple):
                key = ','.join(str(x) for x in k)
            else:
                key = str(k)
            policy[key] = str(v)

        return jsonify({'ok': True, 'policy': policy})

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({'ok': False, 'error': tb})


if __name__ == '__main__':
    print("=" * 60)
    print("  🚁  MDP Drone Challenge — Backend Server")
    print("=" * 60)
    print(f"  Serving on http://localhost:5050")
    print(f"  Edit policy.py then click ▶ Run in the browser")
    print("=" * 60)

    try:
        from waitress import serve

        serve(app, host='127.0.0.1', port=5050, threads=4)
    except ImportError:
        app.run(host='127.0.0.1', port=5050, debug=False, use_reloader=False)
