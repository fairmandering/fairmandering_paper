#!/usr/bin/env python
# coding: utf-8

# In[1]:
from IPython import get_ipython

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from gerrypy.paper.acda.all_states import *
import os
import gerrypy.constants


# In[3]:


fig_dir = os.path.join(constants.RESULTS_PATH, "PNAS", "figures")
si_fig_dir = os.path.join(constants.RESULTS_PATH, "PNAS", "figures", "si_figures")
ensemble_dir = os.path.join(constants.RESULTS_PATH,
                            "allstates", "aaai_columns1595813019", "pnas_results")
ensemble_column_path = os.path.join(constants.RESULTS_PATH,
                            "allstates", "aaai_columns1595813019")
house_results_path = os.path.join(constants.GERRYPY_BASE_PATH, "data",
                                  "misc", "1976-2018-house2.csv")
result_results_dirs = [
    "uniform_random_columns1606420958",
    "uniform_random_columns1606421299",
    "uniform_random_columns1606421397",
    "uniform_random_columns1606421542"
]


# In[4]:


affiliation_df = make_affiliation_df()


# In[5]:


plot_state_affiliation(affiliation_df, si_fig_dir, 3)


# In[6]:


state_election_summary_df = make_state_election_table()


# In[7]:


pd.set_option('display.max_colwidth', 500)
print(state_election_summary_df.to_latex(column_format='lllp{10cm}'))


# In[8]:


state_graph_table = make_state_graph_table()
print(state_graph_table.to_latex())


# In[9]:


ensemble_results = load_all_state_results(ensemble_dir)


# In[10]:


mean_affiliation = affiliation_df.iloc[2]
seat_share_box_df = create_seat_share_box_df(ensemble_results, mean_affiliation)


# In[11]:


seat_dict = {s: {'R': r['r_advantage']['objective_value'],
                  'D': constants.seats[s]['house'] - r['d_advantage']['objective_value']} 
                                        for s, r in ensemble_results.items()}


# In[12]:


pd.DataFrame(seat_dict).sum(axis=1)


# In[13]:


mean_affiliation = affiliation_df.iloc[2]
winner_df = load_historical_house_winner_df(house_results_path, 2010)
seat_fractions = (winner_df.groupby(['year', 'state_po']).apply(
        lambda x: sum(x == 'republican') / constants.seats[x.name[1]]['house'])
    ).unstack('state_po').mean(axis=0)
seat_share_box_df = create_seat_share_box_df(ensemble_results, mean_affiliation)
plot_seat_share_distribution(fig_dir, seat_share_box_df,
                             mean_affiliation, seat_fractions)


# In[14]:


feas = responsiveness_to_feasibility(ensemble_results, mean_affiliation, 0, 5)
domain, state_feasible, seats_feasible = feas
plot_feasibility_by_responsiveness(fig_dir, ensemble_results, mean_affiliation)


# In[15]:


seat_df_by_year = winner_df.unstack('year')
seat_change_dict = {}
for year, next_year in zip(seat_df_by_year.columns[:-1], seat_df_by_year.columns[1:]):
    seat_change_dict[str(year) + '-' + str(next_year)] = (seat_df_by_year[year] != seat_df_by_year[next_year]).groupby('state_po').sum()
seat_series = pd.Series({state: seat_dict['house'] for state, seat_dict
                         in constants.seats.items()})
competitive_box_df = create_competitiveness_box_df(ensemble_results, seat_series)
plot_competitiveness_distribution(si_fig_dir, ensemble_results, seat_change_dict)


# In[16]:


spearman_df = compute_fairness_compactness_correlations(ensemble_results, mean_affiliation)
print(correlation_table(spearman_df).to_latex())


# In[17]:


plot_fairness_correlation(fig_dir, spearman_df, mean_affiliation)


# In[18]:


try:
    state_compactness = pickle.load(open('state_compactness.p', 'rb'))
except FileNotFoundError:  
    state_compactness = compute_historical_compactness(ensemble_results)
    pickle.dump(state_compactness, open('state_compactness.p', 'wb'))
historical_dispersion, historical_roeck, historical_cut_edges = state_compactness


# In[19]:


cut_edges_box_df = create_compactness_box_df(ensemble_results, 'cut_edges', historical_cut_edges)


# In[20]:


plot_centralization_distribution(si_fig_dir, ensemble_results, historical_dispersion, min_seats=3)


# In[21]:


plot_roeck_distribution(si_fig_dir, ensemble_results, historical_roeck, min_seats=3)


# In[22]:


plot_cut_edges_distributions(fig_dir, ensemble_results, historical_cut_edges, min_seats=3)


# In[23]:


ensemble_table = make_ensemble_parameter_table(ensemble_column_path)


# In[24]:


print(ensemble_table.to_latex())


# In[25]:


random_ensemble_results = {}
for dir_slice in result_results_dirs:
    path = os.path.join(constants.RESULTS_PATH, "allstates", dir_slice, "pnas_results")
    for file in os.listdir(path):
        random_ensemble_results[file[:2]] = pickle.load(open(os.path.join(path, file), 'rb'))


# In[26]:


random_cols_seat_share_df = create_seat_share_box_df(random_ensemble_results, mean_affiliation)


# In[27]:


plot_seat_share_ensemble_comparison(random_cols_seat_share_df, seat_share_box_df, 
                                    si_fig_dir, seat_fractions)


# In[28]:


cut_edges_box_df = create_compactness_box_df(ensemble_results, 'cut_edges', historical_cut_edges)
random_centers_cut_edges_box_df = create_compactness_box_df(random_ensemble_results, 'cut_edges', historical_cut_edges)


# In[29]:


plot_compactness_ensemble_comparison(random_centers_cut_edges_box_df, cut_edges_box_df, 
                                    si_fig_dir, historical_cut_edges)


# In[30]:


runtime_df = create_subproblem_runtime_df(ensemble_column_path)


# In[31]:


len(runtime_df)


# In[32]:


(np.mean(runtime_df.partition_time < .5),
np.mean(runtime_df.partition_time < 1),
np.mean(runtime_df.partition_time < 3),
np.max(runtime_df.partition_time))


# In[33]:


runtime_df.partition_time.sum() / (60**2 * 24)


# In[34]:


plot_partition_runtimes(si_fig_dir, runtime_df)


# In[35]:


master_metrics_dfs = create_master_metrics_dfs(ensemble_dir)


# In[36]:


master_runtime_df = pd.concat(master_metrics_dfs)


# In[37]:


(master_runtime_df.construction_time + master_runtime_df.solve_time).sum()


# In[38]:


36009.81886768341 / 60**2


# In[39]:


master_runtime_df = master_runtime_df.query('n_leaves > 5')


# In[40]:


len(master_runtime_df)


# In[41]:


master_runtime_df.head()


# In[ ]:





# In[42]:


master_runtime_df.n_leaves.max()


# In[43]:


master_runtime_df.query('construction_time > 40')


# In[44]:


(
    np.mean(master_runtime_df.construction_time + master_runtime_df.solve_time < .1),
np.mean(master_runtime_df.construction_time + master_runtime_df.solve_time < .5),
 np.mean(master_runtime_df.construction_time + master_runtime_df.solve_time < 1),
 np.mean(master_runtime_df.construction_time + master_runtime_df.solve_time < 3),
 np.max(master_runtime_df.construction_time + master_runtime_df.solve_time))


# In[45]:


plot_master_problem_runtimes(si_fig_dir, master_runtime_df)


# In[46]:


plot_master_convergence(si_fig_dir, master_metrics_dfs)


# In[48]:


state_lines, district_lines = create_lower_48_map_lines(ensemble_column_path, ensemble_dir)


# In[49]:


plot_lower_48(state_lines, district_lines, si_fig_dir)


# # Algorithm Config

# In[50]:


nc_generation_path = os.path.join(constants.RESULTS_PATH,
                                  "PNAS", "PNAS_NC_generation_v2_results_1605458455")
il_generation_path = os.path.join(constants.RESULTS_PATH,
                                  "PNAS", "PNAS_IL_generation_v2_results_1605457815")
vary_population_trials_path = os.path.join(constants.RESULTS_PATH,
                                            "PNAS", "PNAS_population_tolerance_results_1600392876")
il_vary_k_path = os.path.join(constants.RESULTS_PATH, 'PNAS', 'PNAS_IL_k_results_1600727522')
nc_vary_k_path = os.path.join(constants.RESULTS_PATH, 'PNAS', 'PNAS_NC_k_results_1600809414')


# In[51]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
from gerrypy.paper.acda.algorithm_configuration import *
get_ipython().run_line_magic('reload_ext', 'autoreload')


# In[52]:


nc_generation_trial_df = load_trials_df(nc_generation_path)
il_generation_trial_df = load_trials_df(il_generation_path)


# In[53]:


generation_df = pd.concat([nc_generation_trial_df, il_generation_trial_df])
formatted_generation_df = process_state_trial_df(generation_df,
                                                 ['State', 'Centers', 'Capacities']).drop(columns='Roeck')
print(formatted_generation_df.to_latex(escape=False))


# In[54]:


print(pd.concat([formatted_generation_df.groupby(['State', 'Centers']).mean(), 
         formatted_generation_df.groupby(['State', 'Capacities']).mean()]).round(4).to_latex(escape=False))


# In[55]:


print(formatted_generation_df.groupby(['State', 'Centers']).mean().to_latex(escape=False))


# In[56]:


population_trials = load_trials_df(vary_population_trials_path)


# In[57]:


process_population_trial_df = process_state_trial_df(population_trials,
                                                      ['State', '$\epsilon_p$']).drop(columns=['Roeck'])
print(process_population_trial_df.to_latex(escape=False))


# In[58]:


distributions = load_seat_distribution_by_epsilon(vary_population_trials_path)


# In[59]:


plot_il_seat_distributions_varying_epislon(si_fig_dir, distributions)


# In[60]:


plot_nc_seat_distributions_varying_epislon(si_fig_dir, distributions)


# In[61]:


il_vary_k_trials = load_trials_df(il_vary_k_path)
nc_vary_k_trials = load_trials_df(nc_vary_k_path)


# In[62]:


il_vary_k_df = process_vary_k_trial_df(il_vary_k_trials)
nc_vary_k_df = process_vary_k_trial_df(nc_vary_k_trials)


# In[63]:


print(pd.concat([nc_vary_k_df, il_vary_k_df]).to_latex(escape=False))


# In[64]:


nc_percentiles = seat_share_with_k_distribution(nc_vary_k_path)
il_percentiles = seat_share_with_k_distribution(il_vary_k_path)


# In[65]:


plot_nc_seat_distribution_varying_k(si_fig_dir, nc_percentiles)


# In[66]:


plot_il_seat_distribution_varying_k(si_fig_dir, il_percentiles)


# In[ ]:





# In[ ]:




