import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all_visualizations(api, value_history, policy, delta_history, THETA):
    env_params = api.get_env_params()
    rows = env_params['rows']
    cols = env_params['cols']

    value_heatmap(api, value_history, rows, cols)

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
                    if api.api._env['startPos'] == {'row': r, 'col': c}:
                        ax.text(c, r, '🚁', ha='center', va='center', fontsize=12)
                    elif api.is_terminal((r, c, dmg, 0)):
                        ax.text(c, r, '🎯', ha='center', va='center', fontsize=12)
                    elif api.api._is_obstacle(r, c):
                        ax.text(c, r, '🧱', ha='center', va='center', fontsize=12)
                    elif api.api._cell(r, c) == 'medkit':
                        ax.text(c, r, '💊', ha='center', va='center', fontsize=12)
                    elif api.api._cell(r, c) in ('portal_a', 'portal_b'):
                        ax.text(c, r, '🌀', ha='center', va='center', fontsize=12)
                    elif api.api._cell(r, c) == 'storm':
                        sev = api.api._storm_sev.get(f'{r},{c}', 1)
                        if sev == 1:
                            ax.text(c, r, '1', ha='center', va='center', fontsize=12)
                        elif sev == 2:
                            ax.text(c, r, '2', ha='center', va='center', fontsize=12)
                        elif sev == 3:
                            ax.text(c, r, '3', ha='center', va='center', fontsize=12)
                    elif api.api.is_in_storm_zone((r, c, dmg)):
                        ax.text(c+.5, r, '🌩️', ha='center', va='center', fontsize=12)
            
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        output_path = os.path.join('visualize', f'value_heatmap_{type}.png')
        plt.tight_layout(h_pad=1.5)
        plt.subplots_adjust(top=0.95)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()