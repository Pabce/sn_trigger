# Script to read a bunch of info output files and, generate some plots/summary statistics,
# and return the best performing configurations

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config as cf

import generate_configs as gc

# # Settings --------------------------------------
# # The info output files location
# info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/test'
# #info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/livermore'
# info_output_file_base_str = 'info_output_random_'
# # The number of files to read
# files_to_read = 3500 # 'all', an integer, or a list/numpy 1d array of integers

# # --------------------------------------

def read_and_pool(info_output_dir, info_output_file_base_str, files_to_read, start_at=0):
    if type(files_to_read) != int:
        num_files_to_read = np.inf
    else:
        num_files_to_read = files_to_read

    # Locate all files in the directory that match the base string
    info_files = [f for f in os.listdir(info_output_dir) if info_output_file_base_str in f]

    # Read the files, parsing the json into a dictionary
    info_data = {}
    count = 0
    for i, file in enumerate(info_files):
        if count >= num_files_to_read:
            break

        # The config path ends in _NUMBER.yaml. Extract the number (can be more than one digit)
        file_number = int(file.split('_')[-1].split('.')[0])
        if file_number < start_at:
            continue

        if (type(files_to_read) == list or type(files_to_read) == np.ndarray) and file_number not in files_to_read:
            continue

        # Read the json into a dictionary
        try:
            with open(os.path.join(info_output_dir, file), 'r') as f:
                data = json.load(f)
                # If 'load_data' or 'clustering' is not in the data, there was an issue in the simulation
                # not related to the clustering parameters (probably some random exception).
                # So we skip the file
                if ('load_events' not in data) or ('clustering' not in data):
                    continue

                info_data[file] = data
                
        except (OSError, json.decoder.JSONDecodeError) as e:
            print(f'Error reading file: {file}')
            print(e)
            continue

        count += 1
        print(f'Read file: {file}, count: {count}')

    # Create a dataframe from the dictionary
    df = pd.DataFrame(info_data).T
    print(df)

    # "Clean" version of the dataframe, removing rows with NaN values in the specified columns
    # This will be the case when the simulation was cut short due to (probably) too few clusters
    df_clean = df.dropna(axis=0, subset=['bdt_training', 'trigger_efficiency'])
    # Also keep the dead points in a separate dataframe
    df_na = df[~df.index.isin(df_clean.index)]

    # Check if the df_na is empty
    if df_na.empty:
        print("No NaN values found in the dataframe")
    else:
        print("NaN values found in the dataframe")

    # Get the list of distances and models used.
    # They should be the same for all files, so we can just take the first one
    model_names = list( df_clean['trigger_efficiency'][0].get('trigger_efficiencies').keys() )
    print(f"Model names: {model_names}")

    # Check if we have 'distance_to_evaluate' or 'number_of_interactions_to_evaluate'
    distance_to_evaluate = df_clean['trigger_efficiency'][0].get('distance_to_evaluate')
    number_of_interactions_to_evaluate = df_clean['trigger_efficiency'][0].get('number_of_interactions_to_evaluate')
    if distance_to_evaluate is not None:
        distances_or_num_interactions = distance_to_evaluate
    else:
        distances_or_num_interactions = number_of_interactions_to_evaluate


    # Get some arrays from the config and the output dicts of the different stages
    config_column = df_clean['config']
    config_expanded = pd.json_normalize(config_column, max_level=2)
    config_expanded.index = config_column.index
    if not df_na.empty:
        config_column_na = df_na['config']
        config_expanded_na = pd.json_normalize(config_column_na, max_level=2)
        config_expanded_na.index = config_column_na.index

    # Cross-check: make sure that you have the correct detector
    # detector_type = config_expanded['Detector.type']
    # print(f"Detector type: {detector_type}")
    # # Print the unique values
    # print(f"Unique values: {config_expanded['Detector.type'].unique()}")

    cparams = config_expanded['Simulation.clustering.parameters'].apply(pd.Series)
    if not df_na.empty:
        cparams_na = config_expanded_na['Simulation.clustering.parameters'].apply(pd.Series)
    #print(f"cparams.columns: {cparams.columns}")

    # TODO: This is likely to cause problems when there are multiple models -----------
    # Another stupid expansion to get the pinching parameters
    config_expanded_2 = pd.json_normalize(config_column, max_level=4)
    supernova_spectra = config_expanded_2['Simulation.trigger_efficiency.physics.supernova_spectra'].apply(pd.Series)
    supernova_spectra_expanded = supernova_spectra.join(
        pd.json_normalize(supernova_spectra.iloc[:, 0])
    ).drop(columns=[0])     # remove the original dict column

    supernova_spectra_expanded.index = config_column.index
    # --------------------------------------------------------------------------------

    # Trigger efficiencies and clustering output we only have for the clean dataframe
    trigger_efficiency = df_clean['trigger_efficiency'].apply(pd.Series)
    trigger_efficiencies = trigger_efficiency['trigger_efficiencies'].apply(pd.Series).dropna(axis=0)
    clustering = df_clean['clustering'].apply(pd.Series)

    # plt.scatter(cparams['max_cluster_time'], trigger_efficiencies['GKVM model'].str[2])
    # plt.show()

    # Put everything together in a single dataframe for each model
    opdf = {}
    if not df_na.empty:
        opdf_na = cparams_na
        opdf_na['trigger_efficiency'] = -1
    else:
        opdf_na = None

    for model in model_names:
        # Separate the trigger efficiencies into different columns for each distance
        # Drop rows where the list is not complete

        trigger_efficiencies_columns = trigger_efficiencies[model].apply(pd.Series)
        trigger_efficiencies_columns.dropna(axis=0, inplace=True)
        trigger_efficiencies_columns.columns = [f'trigger_efficiency.{d}' for d in distances_or_num_interactions]

        # For the clean dataframe
        opdf[model] = pd.concat([trigger_efficiencies_columns, cparams, supernova_spectra_expanded, clustering, df_clean], axis=1)

        #print(opdf[model])
        print(opdf[model].columns)
    
    return opdf, opdf_na, model_names, distances_or_num_interactions, config_column

# Find the top n configurations with the highest trigger efficiencies for the given models and distances
def find_top_n_configs(n, model_names, distances_or_num_interactions, opdf):
    top_configs_df = {}
    for model in model_names:
        top_configs_df[model] = {}
        for i, distance_or_num_interaction in enumerate(distances_or_num_interactions):

            df_sorted = opdf[model].sort_values(by=f'trigger_efficiency.{distance_or_num_interaction}', ascending=False)

            top_configs_df[model][distance_or_num_interaction] = df_sorted.head(n)

            # Print the top n configurations with all columns
            pd.set_option('display.max_columns', None)
            print(f"Top {n} configurations for model {model}, 'distances_or_num_interactions': {distance_or_num_interaction}")
            print(df_sorted
                .loc[:, [f'trigger_efficiency.{distance_or_num_interaction}', 'max_cluster_time', 'max_hit_time_diff', 'max_hit_distance', 'min_hit_multiplicity', 'min_neighbours']]
                .head(n))
            print("\n")

    return top_configs_df


if __name__=='__main__':
    # Settings --------------------------------------
    # The info output files location
    info_output_file_base_str = 'info_output_random_'
    # The number of files to read
    files_to_read = 3500 # 'all', an integer, or a list/numpy 1d array of integers

    # Do you want to generate the top configs for each model? With distances or number of interactions?
    generate_top_configs_for_each_model = False
    # Do you want to generate the top configs for each distance or number of interactions?
    gen_use_distances = False
    # --------------------------------------

    # Standard VD
    #info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/livermore'
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_standard'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd/base_optimized_vd.yaml'
    # config_output_dir = '../../configs/optimized_vd/'
    # config_output_base_str = 'optimized_vd_'
    # config_list_file_name = 'optimized_vd_config_list.txt'

    # Standard VD CC only
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_cconly'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_cconly/base_optimized_vd_cconly.yaml'
    # config_output_dir = '../../configs/optimized_vd_cconly/'
    # config_output_base_str = 'optimized_vd_cconly_'
    # config_list_file_name = 'optimized_vd_cconly_config_list.txt'

    # Standard VD laronly
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_laronly'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_laronly/base_optimized_vd_laronly.yaml'
    # config_output_dir = '../../configs/optimized_vd_laronly/'
    # config_output_base_str = 'optimized_vd_laronly_'
    # config_list_file_name = 'optimized_vd_laronly_config_list.txt'

    # Low gammas VD
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_lowgammas'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_lowgammas/base_optimized_vd_lowgammas.yaml'
    # config_output_dir = '../../configs/optimized_vd_lowgammas/'
    # config_output_base_str = 'optimized_vd_lowgammas_'
    # config_list_file_name = 'optimized_vd_lowgammas_config_list.txt'

    # Super low gammas VD
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_superlowgammas'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_superlowgammas/base_optimized_vd_superlowgammas.yaml'
    # config_output_dir = '../../configs/optimized_vd_superlowgammas/'
    # config_output_base_str = 'optimized_vd_superlowgammas_'
    # config_list_file_name = 'optimized_vd_superlowgammas_config_list.txt'

    # Low radon VD
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_lowradon'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_lowradon/base_optimized_vd_lowradon.yaml'
    # config_output_dir = '../../configs/optimized_vd_lowradon/'
    # config_output_base_str = 'optimized_vd_lowradon_'
    # config_list_file_name = 'optimized_vd_lowradon_config_list.txt'

    # Random VD RW
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_rw'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_rw/base_optimized_vd_rw.yaml'
    # config_output_dir = '../../configs/optimized_vd_rw/'
    # config_output_base_str = 'optimized_vd_rw_'
    # config_list_file_name = 'optimized_vd_rw_config_list.txt'

    # High efficiency VD
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_higheff'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_higheff/base_optimized_vd_higheff.yaml'
    # config_output_dir = '../../configs/optimized_vd_higheff/'
    # config_output_base_str = 'optimized_vd_higheff_'
    # config_list_file_name = 'optimized_vd_higheff_config_list.txt'

    # Random VD nowall
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_nowall'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_nowall/base_optimized_vd_nowall.yaml'
    # config_output_dir = '../../configs/optimized_vd_nowall/'
    # config_output_base_str = 'optimized_vd_nowall_'
    # config_list_file_name = 'optimized_vd_nowall_config_list.txt'

    # High ADC VD
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_highadc'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_highadc/base_optimized_vd_highadc.yaml'
    # config_output_dir = '../../configs/optimized_vd_highadc/'
    # config_output_base_str = 'optimized_vd_highadc_'
    # config_list_file_name = 'optimized_vd_highadc_config_list.txt'

    # Standard HD
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/hd_standard'
    # starting_config_number = 0
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_hd/base_optimized_hd.yaml'
    # config_output_dir = '../../configs/optimized_hd/'
    # config_output_base_str = 'optimized_hd_'
    # config_list_file_name = 'optimized_hd_config_list.txt'

    # -------------------------------------
    # Standard VD CC only SN parameter grid
    # info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger/vd_cconly'
    # starting_config_number = 3200
    # delete_old = True
    # base_config_file_path = '../../configs/optimized_vd_cconly/base_optimized_vd_cconly.yaml'
    # config_output_dir = '../../configs/optimized_vd_cconly_sn_parameter_grid/'
    # config_output_base_str = 'optimized_vd_cconly_sn_parameter_grid_'
    # config_list_file_name = 'optimized_vd_cconly_sn_parameter_grid_config_list.txt'
    # GEN_SN_PARAMETER_GRID = True
    # generate_top_configs_for_each_model = True
    # READ_SN_PARAMETER_GRID = False

    # -------------------------------------
    # # READING the SN PARAMETER GRID OUTPUTS
    info_output_dir = '/eos/project-e/ep-nu/pbarhama/trigger-grid/vd_cconly'
    info_output_file_base_str = 'info_output_grid_'
    files_to_read = 5000
    READ_SN_PARAMETER_GRID = True
    GEN_SN_PARAMETER_GRID = False
    generate_top_configs_for_each_model = False


    opdf, opdf_na, model_names, distances_or_num_interactions, configs_col = read_and_pool(info_output_dir, info_output_file_base_str,
                                                                      files_to_read, start_at=0)
    if READ_SN_PARAMETER_GRID:
        # Save the opdf to a pickle file in local_saves
        import pickle
        pickle_jar = [opdf, opdf_na, model_names, distances_or_num_interactions, configs_col]
        with open('../../local_saves/opdf_sn_parameter_grid.pkl', 'wb') as f:
            pickle.dump(pickle_jar, f)

        
        print(model_names)
        print(distances_or_num_interactions)
        print(opdf[model_names[0]].keys())

        model = model_names[0]

        for num_events in distances_or_num_interactions:

            print(f"Num events: {num_events}")
            # plt.figure()
            # plt.scatter(opdf['GKVM model']['max_cluster_time'], opdf['GKVM model'][f'trigger_efficiency.{distance}'], s=20)
            # plt.scatter(opdf_na['max_cluster_time'], np.zeros(opdf_na.shape[0])-1, c='red', s=15)

            fig, ax = plt.subplots()
            opdf[model].plot(x='pinching_parameters.average_energy', y='pinching_parameters.alpha', kind='scatter', 
                             ax=ax, c=f'trigger_efficiency.{num_events}', cmap='gnuplot2')
            ax.set_title(f"{model}, Num events: {num_events}")

            # fig, ax = plt.subplots()
            # opdf[model].plot(x='max_cluster_time', y=f'trigger_efficiency.{distance}', kind='scatter', ax=ax)
            # ax.set_title(f"{model}, Distance: {distance} kpc")


            # fig, ax = plt.subplots()
            # opdf[model].plot(kind='scatter', x='sn_eff_clusters_num', y='bg_eff_clusters_num',
            #                     c=f'trigger_efficiency.{distance}', cmap='inferno', ax=ax)
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # ax.set_title(f"{model}, Distance: {distance} kpc")
        
            plt.show()
        
        exit('testing sn pooling')

    # --------------------------------------------    

    print(f'Model names: {model_names}')
    if gen_use_distances:
        print(f'Distances: {distances_or_num_interactions}')
    else:
        print(f'Number of interactions: {distances_or_num_interactions}')

    top_configs_df = find_top_n_configs(3, model_names, distances_or_num_interactions, opdf)

    # Do you want to generate the top configs for each model? With distances or number of interactions?
    if not generate_top_configs_for_each_model:
        exit()
    
    if not GEN_SN_PARAMETER_GRID:
        new_distances_per_model = {}
        new_num_interactions_per_model = {}
        if gen_use_distances:
            new_distances_per_model['Livermore'] = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 29.0, 32.0, 35.0, 38.0]
            new_distances_per_model['GKVM'] = [11.0, 14.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 32.0, 35.0, 40.0, 45.0, 50.0]
            new_distances_per_model['Garching'] = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 29.0, 32.0, 35.0, 38.0]
            new_distances_per_model['Nakazato30'] = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 29.0, 32.0, 35.0, 38.0]
        else:
            new_num_interactions_per_model['Livermore'] = [20, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 500, 1000]
            new_num_interactions_per_model['GKVM'] = [20, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 500, 1000]
            new_num_interactions_per_model['Garching'] = [20, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 500, 1000]
            new_num_interactions_per_model['Nakazato30'] = [20, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 500, 1000]

        # Generate the top configurations
        gc.generate_top_n_configs(top_configs_df, configs_col, model_names, distances_or_num_interactions,
                                    starting_config_number, delete_old, base_config_file_path,
                                    config_output_dir, config_output_base_str, config_list_file_name,
                                    new_distances_per_model, new_num_interactions_per_model)

    else:
        new_distances_per_model = {}
        new_num_interactions_per_model = {}

        new_num_interactions_per_model['Livermore'] = [10, 30, 45, 60, 75, 90, 105, 120, 150, 180, 240, 300, 360, 500]

        supernova_parameter_grid = {
            'average_energy': {'mode': 'uniform', 'range': [8, 18]},
            'alpha': {'mode': 'uniform', 'range': [1.4, 3.2]},
        }
        n_configs = 300
        model_to_select = 'Livermore'
        distance_or_num_interactions = 90

        gc.generate_sn_parameter_grid_random_configs(n_configs, top_configs_df, configs_col, 'Livermore', 'Custom', 
                                                     distance_or_num_interactions, starting_config_number, delete_old, 
                                                     base_config_file_path, config_output_dir, config_output_base_str, 
                                                     config_list_file_name, new_distances_per_model, new_num_interactions_per_model, 
                                                     supernova_parameter_grid)
        
        # def generate_sn_parameter_grid_random_configs(top_configs_df, configs_col, model_to_select, model_to_label, distances_or_num_interactions,
        #                         starting_config_number, delete_old, base_config_file_path, 
        #                         config_output_dir, config_output_base_str, 
        #                         config_list_file_name, new_distances_per_model, new_num_interactions_per_model,
        #                         supernova_parameter_grid):
        
    exit()

    # Create config files for the top configurations
    # base_config_path = '../../configs/optimized_vd/base_optimized_vd.yaml'
    # bdt_hyperparameters = {}
    # for model in model_names:
    #     bdt_hyperparameters[model] = {}
    #     for distance in distances:
    #         for i, index in enumerate(top_configs_df[model][distance].index):
    #             info_output = top_configs_df[model][distance].loc[index]
    #             config = configs_col.loc[index]
                
    #             clustering_parameters = config['Simulation']['clustering']['parameters']
    #             #bdt_hyperparameters[model][distance] = config['bdt_training']['bdt_hyperparameters']
    #             bdt_hyperparameters[model][distance] = info_output['bdt_training']['bdt_params']

    #             print(f"\nModel: {model}, Distance: {distance}, Index: {index}")
    #             #print(bdt_hyperparameters[model][distance])

    #             # Create a new config file with the bdt parameters and the clustering parameters

    #             output_file_name = f'optimized_vd_{model}_{distance}_{i}.yaml'
    #             # Remove whitespace from the output file name (TEMPORARY FIX)
    #             output_file_name = output_file_name.replace(' ', '_')

    #             new_config_output_path = f'../../configs/optimized_vd/{output_file_name}'
    #             parameters = {}
    #             parameters.update(clustering_parameters)
    #             parameters['optimize_hyperparameters'] = False
    #             parameters['bdt_hyperparameters'] = bdt_hyperparameters[model][distance]
    #             parameters['distance_to_evaluate'] = distance
    #             # parameters['model_name'] = model # TEMP FIX
    #             parameters['model_name'] = model.split(' ')[0] # TEMP FIX

    #             gc.write_config(base_config_path, new_config_output_path, parameters)

    for model in ['Livermore model', 'GKVM model']:
        break
        for distance in distances:

            print(f"Distance: {distance}")
            # plt.figure()
            # plt.scatter(opdf['GKVM model']['max_cluster_time'], opdf['GKVM model'][f'trigger_efficiency.{distance}'], s=20)
            # plt.scatter(opdf_na['max_cluster_time'], np.zeros(opdf_na.shape[0])-1, c='red', s=15)

            fig, ax = plt.subplots()
            opdf[model].plot(x='max_cluster_time', y=f'trigger_efficiency.{distance}', kind='scatter', ax=ax)
            opdf_na.plot(x='max_cluster_time', y='trigger_efficiency', kind='scatter', c='red', ax=ax)
            ax.set_title(f"{model}, Distance: {distance} kpc")

            fig, ax = plt.subplots()
            opdf[model].plot(kind='scatter', x='sn_eff_clusters_num', y='bg_eff_clusters_num',
                             c=f'trigger_efficiency.{distance}', cmap='inferno', ax=ax)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"{model}, Distance: {distance} kpc")
        
            plt.show()
    
    #distance = 25.0
    # sns.pairplot(opdf['GKVM model'][[f'trigger_efficiency.{distance}', 'max_cluster_time', 'max_hit_time_diff',
    #                                   'max_hit_distance', 'min_hit_multiplicity',
    #                                   'min_neighbours', 'sn_eff_clusters_num',
    #                                   'bg_eff_clusters_num']], hue=f'trigger_efficiency.{distance}')
    
    # plt.show()



# filtered_dataf = dataf[min_hit_multiplicities == 12]
# filtered_dataf_np = filtered_dataf[[
#     'max_cluster_times',
#     'max_hit_time_diffs',
#     'max_hit_distances',
#     'min_hit_multiplicities',
#     'min_neighbours',
#     'trigger_efficiencies',
#     'sn_eff_clusters_num',
#     'bg_eff_clusters_num',
# ]].values


# plt.figure()
# plt.scatter(bg_eff_clusters_num, trigger_efficiencies, s=20, c=min_hit_multiplicities)
# plt.title('Background clusters vs trigger efficiencies')

# plt.figure()
# plt.scatter(sn_eff_clusters_num, trigger_efficiencies, s=20)
# plt.title('Signal clusters vs trigger efficiencies')

# plt.figure()
# plt.scatter(sn_eff_clusters_num[trigger_efficiencies >= 0], bg_eff_clusters_num[trigger_efficiencies >= 0], 
#             c=trigger_efficiencies[trigger_efficiencies >= 0], cmap='inferno', s=15)
# plt.colorbar()
# plt.xlabel('Signal clusters')
# plt.ylabel('Background clusters')
# plt.xscale('log')
# plt.yscale('log')

# plt.figure()
# plt.scatter((sn_eff_clusters_num/bg_eff_clusters_num)[trigger_efficiencies >= 0], trigger_efficiencies[trigger_efficiencies >= 0], s=20)
# plt.title('Signal/Background clusters vs trigger efficiencies')

# # plt.figure()
# # plt.scatter(max_cluster_times, trigger_efficiencies)
# # plt.title('Max cluster times vs trigger efficiencies')

# # plt.figure()
# # plt.scatter(max_hit_time_diffs, trigger_efficiencies)
# # plt.title('Max hit time diffs vs trigger efficiencies') 

# plt.figure()
# plt.scatter(max_cluster_times, max_hit_time_diffs, c=trigger_efficiencies, cmap='bwr')
# plt.colorbar()
# plt.tight_layout()
# plt.title('Max cluster times vs max hit time diffs')
# plt.xlabel('Max cluster times')
# plt.ylabel('Max hit time diffs')

# plt.figure()
# plt.scatter(max_cluster_times[trigger_efficiencies>=0], max_hit_distances[trigger_efficiencies>=0], c=trigger_efficiencies[trigger_efficiencies >= 0], cmap='bwr')
# plt.colorbar()
# plt.tight_layout()
# plt.title('Max cluster times vs max hit distances')
# plt.xlabel('Max cluster times')
# plt.ylabel('Max hit distances')

# plt.figure()
# plt.scatter(max_hit_distances, trigger_efficiencies, c=min_hit_multiplicities, cmap='gnuplot')
# plt.title('Max hit distances vs trigger efficiencies')
# plt.xlabel('Max hit distances')
# plt.ylabel('Trigger efficiencies')

# plt.figure()
# plt.scatter(min_hit_multiplicities, trigger_efficiencies)
# plt.title('Min hit multiplicities vs trigger efficiencies')

# # plt.figure()
# # plt.scatter(min_neighbours, trigger_efficiencies)
# # plt.title('Min neighbours vs trigger efficiencies')

# #plt.show()

# # --------------------------------------


# # sns.pairplot(dataf, hue='min_neighbours')
# # plt.show()

# #umap_dataf = dataf[dataf.min_neighbours == 1]
# umap_dataf = dataf
# data_np = umap_dataf[[
#     'max_cluster_times',
#     'max_hit_time_diffs',
#     'max_hit_distances',
#     'min_hit_multiplicities',
# ]].values

# # data_np = dataf[[
# #     'max_cluster_times',
# #     'max_hit_time_diffs',
# #     'max_hit_distances',
# #     'min_hit_multiplicities',
# #     'min_neighbours',
# # ]].values

# print(type(data_np))
# print(data_np.shape)

# reducer = umap.UMAP()
# scaled_data = StandardScaler().fit_transform(data_np)

# embedding = reducer.fit_transform(scaled_data)
# print(embedding.shape)

# plt.figure()
# plt.scatter(embedding[:, 0], embedding[:, 1], c=umap_dataf['trigger_efficiencies'], s=30, cmap='bwr', edgecolor='black')
# plt.colorbar()
# # Plot a star for the BEST configuration (highest trigger efficiency, just ONE)
# plt.scatter(embedding[top_configs.index[0], 0], embedding[top_configs.index[0], 1], marker='*', s=200, c='yellow', edgecolor='black')

# plt.show()
