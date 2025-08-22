# Script to generate random configurations for the Condor job submission
# In particular, we will draw random clustering parameter values from a given range

import numpy as np
import os
import argparse

# Function to generate an arbitrary config file based on a base file and a dictionary of parameters
def write_config(base_config_file_path, new_config_file_path, parameters):
    # Open the base file
    with open(base_config_file_path, 'r') as f1:
        lines = f1.readlines()

        # Write the configuration to the new file
        with open(new_config_file_path, 'w') as f2:
            for line in lines:
                for key, value in parameters.items():
                    if f'**{key}**' in line:
                    
                        # If a dictionary, replace in a yaml format
                        if isinstance(value, dict):
                            line = line.replace(f'**{key}**', '')
                            # Find the number of leading spaces
                            leading_spaces = len(line) - len(line.lstrip(' '))
                            # Add one new line
                            line += '\n'
                            for k, v in value.items():
                                # Add two extra leading spaces to the line
                                line += ' ' * (leading_spaces + 2)
                                # If the value is None, print as null
                                if v is None:
                                    line += f'{k}: null\n'
                                # If the value is a float, always print in decimal format beacuse pyYAML is stupid
                                elif isinstance(v, float):
                                    line += f'{k}: {v:.15f}\n'
                                else:
                                    line += f'{k}: {v}\n'

                        # Otherwise, replace as a string
                        else:
                            line = line.replace(f'**{key}**', str(value))
                            break

                f2.write(line)
        
        print(f'Configuration written to {new_config_file_path}')

def delete_old_configs(config_output_dir, config_output_base_str):
    print('Removing old config files...')
    for file in os.listdir(config_output_dir):
        if file.startswith(config_output_base_str):
            os.remove(os.path.join(config_output_dir, file))

# Function to generate random configurations based on a parameter grid of clustering parameters
def generate_random_configs(n_configs, starting_config_number, delete_old, base_config_file_path, 
                            clustering_parameter_grid, config_output_dir, config_output_base_str, 
                            config_list_file_name):
    
    # Absolute path to the list file
    config_list_file = config_output_dir + config_list_file_name
    print(config_list_file, "FSAJHSDÑKL")

    # Clear the list file
    with open(config_list_file, 'w') as f:
        pass
    # Remove any existing config files if required
    if delete_old:
        delete_old_configs(config_output_dir, config_output_base_str)

    for i in range(n_configs):
        config_number = starting_config_number + i
        parameters = {}
        config_file_path = config_output_dir + f'{config_output_base_str}{config_number}.yaml'

        # Here we train the BDT, so make optimize_hyperparameters True and bdt_parameters None
        parameters['optimize_hyperparameters'] = True
        parameters['bdt_hyperparameters'] = None

        # Sample the clustering parameters
        for key, value in clustering_parameter_grid.items():
            if value['mode'] == 'uniform':
                param_value = np.random.uniform(value['range'][0], value['range'][1])
            elif value['mode'] == 'grid':
                param_value = np.random.choice(value['values'])
            elif value['mode'] == 'uniform-constrained':
                max_value = parameters[value['constraint']]
                param_value = np.random.uniform(value['range'][0], min(value['range'][1], max_value))
                # param_value = np.random.uniform(value['range'][0], value['range'][1])
                # if param_value > max_value:
                #     param_value = max_value

            parameters[key] = param_value

        # Write the configuration to the new file
        write_config(base_config_file_path, config_file_path, parameters)

        # Write the (absolute path) config file name to the list file
        absolute_config_file_path = os.path.abspath(config_file_path)
        print(f'Configuration {config_number} ({i+1}/{n_configs}) written to {absolute_config_file_path}')
        with open(config_list_file, 'a') as f:
            f.write(absolute_config_file_path + '\n')


# Function to generate configs based on the best performing configurations
def generate_top_n_configs(top_configs_df, configs_col, model_names, distances_or_num_interactions,
                                starting_config_number, delete_old, base_config_file_path, 
                                config_output_dir, config_output_base_str, 
                                config_list_file_name, new_distances_per_model, new_num_interactions_per_model):
    # Absolute path to the list file
    config_list_file = config_output_dir + config_list_file_name

    # distances or num_interactions?
    if new_distances_per_model:
        distances_or_num_interactions_str = 'distance'
    elif new_num_interactions_per_model:
        distances_or_num_interactions_str = 'num_interactions'
    else:
        raise ValueError("No distances or num_interactions provided")

    # Clear the list file
    with open(config_list_file, 'w') as f:
        pass
    # Remove any existing config files if required
    if delete_old:
        delete_old_configs(config_output_dir, config_output_base_str)

    for model in model_names:
        for d in distances_or_num_interactions:
            for i, index in enumerate(top_configs_df[model][d].index):
                config_number = starting_config_number + i

                info_output = top_configs_df[model][d].loc[index]
                config = configs_col.loc[index]
                
                clustering_parameters = config['Simulation']['clustering']['parameters']

                print(f"\nModel: {model}, {distances_or_num_interactions_str}: {d}, Index: {index}")

                # Create a new config file with the bdt parameters and the clustering parameters

                config_file_path = config_output_dir + f'{config_output_base_str}{model}_{distances_or_num_interactions_str}_{d}_{i}.yaml'
                # Remove whitespace from the output file name (TEMPORARY FIX)
                config_file_path = config_file_path.replace(' ', '_')

                parameters = {}
                parameters.update(clustering_parameters)
                parameters['optimize_hyperparameters'] = False
                # TEMP FIX
                try:
                    parameters['bdt_hyperparameters'] = info_output['bdt_training']['bdt_hyperparameters']
                except KeyError: # This is for older info outputs
                    parameters['bdt_hyperparameters'] = info_output['bdt_training']['bdt_params']

                if distances_or_num_interactions == 'distance':
                    parameters['distance_to_evaluate'] = new_distances_per_model[model]
                    parameters['number_of_interactions_to_evaluate'] = 'null'
                else:
                    parameters['number_of_interactions_to_evaluate'] = new_num_interactions_per_model[model]
                    parameters['distance_to_evaluate'] = 'null'
                
                # Get the model parameters
                parameters['label'] = model

                # Search for the model that contains the string label: <model>
                model_index = [i for i, s in enumerate(info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra']) if s['label']==model][0]

                # Get the model parameters
                parameters['average_energy'] = info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra'][model_index]['pinching_parameters']['average_energy']
                parameters['alpha'] = info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra'][model_index]['pinching_parameters']['alpha']
                parameters['num_interactions_cc'] = info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra'][model_index]['interaction_number_10kpc']['cc']
                parameters['num_interactions_es'] = info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra'][model_index]['interaction_number_10kpc']['es']

                write_config(base_config_file_path, config_file_path, parameters)

                # Write the (absolute path) config file name to the list file
                absolute_config_file_path = os.path.abspath(config_file_path)
                print(f'Configuration {config_number} written to {absolute_config_file_path}')
                with open(config_list_file, 'a') as f:
                    f.write(absolute_config_file_path + '\n')


# Function to generate configs sampling the <E, alpha> parameter space, given an optimal configuration
def generate_sn_parameter_grid_random_configs(n_configs, top_configs_df, configs_col, model_to_select, model_to_label, 
                                distance_or_num_interactions,
                                starting_config_number, delete_old, base_config_file_path, 
                                config_output_dir, config_output_base_str, 
                                config_list_file_name, new_distances_per_model, new_num_interactions_per_model,
                                supernova_parameter_grid):

    # Absolute path to the list file
    config_list_file = config_output_dir + config_list_file_name

    # distances or num_interactions?
    if new_distances_per_model:
        distances_or_num_interactions_str = 'distance'
    elif new_num_interactions_per_model:
        distances_or_num_interactions_str = 'num_interactions'
    else:
        raise ValueError("No distances or num_interactions provided")

    # Clear the list file
    with open(config_list_file, 'w') as f:
        pass
    # Remove any existing config files if required
    if delete_old:
        delete_old_configs(config_output_dir, config_output_base_str)

    model = model_to_select

    top_config_index = top_configs_df[model][distance_or_num_interactions].index[0]
    top_config = top_configs_df[model][distance_or_num_interactions].loc[top_config_index]
    info_output = top_configs_df[model][distance_or_num_interactions].loc[top_config_index]
    config = configs_col.loc[top_config_index]
    

    for i in range(n_configs):

        config_number = starting_config_number + i
        clustering_parameters = config['Simulation']['clustering']['parameters']

        print(f"\nModel: {model_to_label}, {distances_or_num_interactions_str}: {distance_or_num_interactions}, Index: {top_config_index}")

        # Create a new config file with the bdt parameters and the clustering parameters
        config_file_path = config_output_dir + f'{config_output_base_str}{model_to_label}_{distances_or_num_interactions_str}_{config_number}.yaml'
        # Remove whitespace from the output file name (TEMPORARY FIX)
        config_file_path = config_file_path.replace(' ', '_')

        parameters = {}
        parameters.update(clustering_parameters)
        parameters['optimize_hyperparameters'] = False
        parameters['bdt_hyperparameters'] = info_output['bdt_training']['bdt_hyperparameters']

        if distances_or_num_interactions_str == 'distance':
            parameters['distance_to_evaluate'] = new_distances_per_model[model]
            parameters['number_of_interactions_to_evaluate'] = 'null'
        else:
            parameters['number_of_interactions_to_evaluate'] = new_num_interactions_per_model[model]
            parameters['distance_to_evaluate'] = 'null'
        
        # Set the 'custom model' label
        parameters['label'] = model_to_label

        # Sample the supernova parameters

        for key, value in supernova_parameter_grid.items():
            if value['mode'] == 'uniform':
                param_value = np.random.uniform(value['range'][0], value['range'][1])
            elif value['mode'] == 'grid':
                param_value = np.random.choice(value['values'])

            parameters[key] = param_value
        
        # And add these placeholder values
        # Search for the model that contains the string label: <model>
        model_index = [i for i, s in enumerate(info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra']) if s['label']==model][0]
        parameters['num_interactions_cc'] = info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra'][model_index]['interaction_number_10kpc']['cc']
        parameters['num_interactions_es'] = info_output['config']['Simulation']['trigger_efficiency']['physics']['supernova_spectra'][model_index]['interaction_number_10kpc']['es']

        write_config(base_config_file_path, config_file_path, parameters)

        # Write the (absolute path) config file name to the list file
        absolute_config_file_path = os.path.abspath(config_file_path)
        print(f'Configuration {config_number} written to {absolute_config_file_path}')
        with open(config_list_file, 'a') as f:
            f.write(absolute_config_file_path + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_configs", type=int, help="Number of random configurations to generate")

    args = parser.parse_args()

    if args.random_configs:
        n_configs = args.random_configs
    else:
        exit('Please specify the number of random configurations to generate with --random_configs')

    # Number of configurations to generate
    #n_configs = 1500
    starting_config_number = 10000
    delete_old = True
    # Random seed
    np.random.seed(None)

    # Standard VD
    # The base config file to modify
    # base_config_file = '../../configs/random_vd/base_random_vd.yaml'
    # # The config output files destination
    # config_output_dir = '../../configs/random_vd/'
    # config_output_base_str = 'random_vd_'
    # config_list_file_name = 'random_vd_config_list.txt'

    # # Standard VD CC only
    # base_config_file = '../../configs/random_vd_cconly/base_random_vd_cconly.yaml'
    # config_output_dir = '../../configs/random_vd_cconly/'
    # config_output_base_str = 'random_vd_cconly_'
    # config_list_file_name = 'random_vd_cconly_config_list.txt'

    # # Standard VD CC only long
    # base_config_file = '../../configs/random_vd_cconly_long/base_random_vd_cconly_long.yaml'
    # config_output_dir = '../../configs/random_vd_cconly_long/'
    # config_output_base_str = 'random_vd_cconly_long_'
    # config_list_file_name = 'random_vd_cconly_long_config_list.txt'

    # # Standard VD laronly
    # base_config_file = '../../configs/random_vd_laronly/base_random_vd_laronly.yaml'
    # config_output_dir = '../../configs/random_vd_laronly/'
    # config_output_base_str = 'random_vd_laronly_'
    # config_list_file_name = 'random_vd_laronly_config_list.txt'

    # Low gammas VD
    # base_config_file = '../../configs/random_vd_lowgammas/base_random_vd_lowgammas.yaml'
    # config_output_dir = '../../configs/random_vd_lowgammas/'
    # config_output_base_str = 'random_vd_lowgammas_'
    # config_list_file_name = 'random_vd_lowgammas_config_list.txt'

    # Super low gammas VD
    # base_config_file = '../../configs/random_vd_superlowgammas/base_random_vd_superlowgammas.yaml'
    # config_output_dir = '../../configs/random_vd_superlowgammas/'
    # config_output_base_str = 'random_vd_superlowgammas_'
    # config_list_file_name = 'random_vd_superlowgammas_config_list.txt'

    # Low radon VD
    # base_config_file = '../../configs/random_vd_lowradon/base_random_vd_lowradon.yaml'
    # config_output_dir = '../../configs/random_vd_lowradon/'
    # config_output_base_str = 'random_vd_lowradon_'
    # config_list_file_name = 'random_vd_lowradon_config_list.txt'

    # High efficiency VD
    # base_config_file = '../../configs/random_vd_higheff/base_random_vd_higheff.yaml'
    # config_output_dir = '../../configs/random_vd_higheff/'
    # config_output_base_str = 'random_vd_higheff_'
    # config_list_file_name = 'random_vd_higheff_config_list.txt'

    # Random VD RW
    # base_config_file = '../../configs/random_vd_rw/base_random_vd_rw.yaml'
    # config_output_dir = '../../configs/random_vd_rw/'
    # config_output_base_str = 'random_vd_rw_'
    # config_list_file_name = 'random_vd_rw_config_list.txt'

    # Random VD nowall
    base_config_file = '../../configs/random_vd_nowall/base_random_vd_nowall.yaml'
    config_output_dir = '../../configs/random_vd_nowall/'
    config_output_base_str = 'random_vd_nowall_'
    config_list_file_name = 'random_vd_nowall_config_list.txt'

    # High ADC VD
    # base_config_file = '../../configs/random_vd_highadc/base_random_vd_highadc.yaml'
    # config_output_dir = '../../configs/random_vd_highadc/'
    # config_output_base_str = 'random_vd_highadc_'
    # config_list_file_name = 'random_vd_highadc_config_list.txt'

    # Standard HD
    # base_config_file = '../../configs/random_hd/base_random_hd.yaml'
    # # The config output files destination
    # config_output_dir = '../../configs/random_hd/'
    # config_output_base_str = 'random_hd_'
    # config_list_file_name = 'random_hd_config_list.txt'


    # File name for the list of produced config files
    # DESIGN DECISION: for now this will be specified in the condor script
    # The info output files destination
    # info_output_file_dir = '../info_outputs/'
    # info_output_file_base_str = 'output_info_random_vd_'

    # The clustering parameters
    # TODO: Add support for other clustering algorithms
    clustering_parameter_grid = {
        'max_cluster_time': {'mode': 'uniform', 'range': [0.01, 0.7]},
        'max_hit_time_diff': {'mode': 'uniform-constrained', 'range': [0.01, 0.7], 'constraint': 'max_cluster_time'},
        'max_hit_distance': {'mode': 'uniform', 'range': [500, 2500]}, # VD, make smaller for HD
        'min_hit_multiplicity': {'mode': 'grid', 'values': [6, 7, 8, 9, 10, 11, 12, 13]},
        'min_neighbours': {'mode': 'grid', 'values': [1, 2]}
    }

    generate_random_configs(n_configs, starting_config_number, delete_old, base_config_file,
                            clustering_parameter_grid, config_output_dir, config_output_base_str, 
                            config_list_file_name=config_list_file_name)






