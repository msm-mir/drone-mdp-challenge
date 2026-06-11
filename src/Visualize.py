import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_all_visualizations(api, value_history, policy, delta_history, THETA):
    env_params = api.get_env_params()
    rows = env_params['rows']
    cols = env_params['cols']

    value_heatmap(api, value_history, rows, cols)
    convergence(delta_history, THETA)
    policy_map(api, policy, rows, cols)

def value_heatmap(api, value_history, rows, cols):
    for type, V in value_history.items():
        fig1, axes1 = plt.subplots(5, 1, figsize=(8, 22))
        fig1.suptitle(f'{type.capitalize()} Value Heatmap for each Damage Level', fontsize=16, fontweight='bold')
        
        for dmg in range(5):
            grid_v = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    grid_v[r, c] = V.get((r, c, dmg, 0), 0)
                            
            ax = axes1[dmg]
            ax.set_title(f"Damage = {dmg}")
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            
            im = ax.imshow(grid_v, cmap='coolwarm', origin='upper')
            
            plt.rcParams['font.family'] = ['Segoe UI Emoji']

            for r in range(rows):
                for c in range(cols):
                    if api.server_api._env['startPos'] == {'row': r, 'col': c}:
                        ax.text(c, r, '🚁', ha='center', va='center', fontsize=12)
                    elif api.is_terminal((r, c, dmg, 0)):
                        ax.text(c, r, '🎯', ha='center', va='center', fontsize=12)
                    elif api.server_api._is_obstacle(r, c):
                        ax.text(c, r, '🧱', ha='center', va='center', fontsize=12)
                    elif api.server_api._cell(r, c) == 'medkit':
                        ax.text(c, r, '💊', ha='center', va='center', fontsize=12)
                    elif api.server_api._cell(r, c) in ('portal_a', 'portal_b'):
                        ax.text(c, r, '🌀', ha='center', va='center', fontsize=12)
                    elif api.server_api._cell(r, c) == 'storm':
                        sev = api.server_api._storm_sev.get(f'{r},{c}', 1)
                        if sev == 1:
                            ax.text(c, r, '1', ha='center', va='center', fontsize=12)
                        elif sev == 2:
                            ax.text(c, r, '2', ha='center', va='center', fontsize=12)
                        elif sev == 3:
                            ax.text(c, r, '3', ha='center', va='center', fontsize=12)
                    elif api.server_api.is_in_storm_zone((r, c, dmg)):
                        ax.text(c+.5, r, '🌩️', ha='center', va='center', fontsize=12)
            
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        output_path = os.path.join('visualize', f'value_heatmap_{type}.png')
        plt.tight_layout(h_pad=1.5)
        plt.subplots_adjust(top=0.95)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def convergence(delta_history, THETA):
    for type, delta in delta_history.items():
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(delta) + 1), delta, color='dodgerblue', 
                linewidth=2, label='max |V_new - V_old|')
        plt.axhline(y=THETA, color='crimson', linestyle='--', linewidth=1.5, label=f'Theta ({THETA})')
        
        plt.title(f'{type.capitalize()} Value Iteration Convergence Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration Number', fontsize=11)
        plt.ylabel('Max Delta V (Log Scale)', fontsize=11)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=11)
        
        output_path = os.path.join('visualize', f'convergence_{type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def policy_map(api, policy, rows, cols):
    for type, p in policy.items():
        fig3, axes3 = plt.subplots(5, 1, figsize=(10, 22))
        fig3.suptitle(f'{type.capitalize()} Optimal Policy Map for each Damage Level', 
                      fontsize=14, fontweight='bold')
        
        action_makers = {
            'N': '↑',
            'S': '↓',
            'E': '→',
            'W': '←'
        }
        
        for dmg in range(5):
            ax = axes3[dmg]
            ax.imshow(np.ones((rows, cols)), cmap='gray', vmin=0, vmax=1.1, origin='upper')
            ax.set_title(f'Damage = {dmg}', fontsize=12)
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))

            ax.set_xticks(np.arange(-.5, cols, 1)),
            ax.set_yticks(np.arange(-.5, rows, 1)),
            ax.grid(which='minor', color='darkgray', linestyle='-', linewidth=1)
            
            for r in range(rows):
                for c in range(cols):
                    state = (r, c, dmg)
                    
                    plt.rcParams['font.family'] = ['Segoe UI Emoji']

                    if api.server_api._env['startPos'] == {'row': r, 'col': c}:
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='limegreen', alpha=0.6))
                    elif api.is_terminal((r, c, dmg, 0)):
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='limegreen', alpha=0.6))
                        ax.text(c, r, '🎯', ha='center', va='center', fontsize=12)
                        continue
                    elif api.server_api._is_obstacle(r, c):
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='darkgray', alpha=0.6))
                        ax.text(c, r, '🧱', ha='center', va='center', fontsize=12)
                        continue
                    elif api.server_api._cell(r, c) == 'medkit':
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='darkturquoise', alpha=0.6))
                    elif api.server_api._cell(r, c) in ('portal_a', 'portal_b'):
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='mediumslateblue', alpha=0.6))
                    elif api.server_api._cell(r, c) == 'storm':
                        sev = api.server_api._storm_sev.get(f'{r},{c}', 1)
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='red', alpha=0.6))
                    elif api.server_api.is_in_storm_zone((r, c, dmg)):
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='chocolate', alpha=0.6))
                    
                    act = p.get(state)
                    if act in action_makers:
                        ax.text(c, r, action_makers[act], ha='center', va='center', 
                                color='royalblue', fontsize=18, fontweight='bold')
        
        output_path = os.path.join('visualize', f'policy_map_{type}.png')
        plt.tight_layout(h_pad=2.0)
        plt.subplots_adjust(top=0.95)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()