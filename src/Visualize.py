def plot_all_visualizations(api, value_history, policy, delta_history, THETA):
    env_params = api.get_env_params()
    rows = env_params['rows']
    cols = env_params['cols']