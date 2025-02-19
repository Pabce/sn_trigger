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
def generate_top_n_configs(top_configs_df, configs_col, model_names, distances,
                                starting_config_number, delete_old, base_config_file_path, 
                                config_output_dir, config_output_base_str, 
                                config_list_file_name, new_distances_per_model):
    # Absolute path to the list file
    config_list_file = config_output_dir + config_list_file_name

    # Clear the list file
    with open(config_list_file, 'w') as f:
        pass
    # Remove any existing config files if required
    if delete_old:
        delete_old_configs(config_output_dir, config_output_base_str)

    for model in model_names:
        for distance in distances:
            for i, index in enumerate(top_configs_df[model][distance].index):
                config_number = starting_config_number + i

                info_output = top_configs_df[model][distance].loc[index]
                config = configs_col.loc[index]
                
                clustering_parameters = config['Simulation']['clustering']['parameters']

                print(f"\nModel: {model}, Distance: {distance}, Index: {index}")

                # Create a new config file with the bdt parameters and the clustering parameters

                config_file_path = config_output_dir + f'{config_output_base_str}{model}_{distance}_{i}.yaml'
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
                parameters['distance_to_evaluate'] = new_distances_per_model[model]
                # parameters['model_name'] = model # TEMP FIX
                parameters['model_name'] = model.split(' ')[0] # TEMP FIX

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
    # # The base config file to modify
    # base_config_file = '../../configs/random_vd/base_random_vd.yaml'
    # # The config output files destination
    # config_output_dir = '../../configs/random_vd/'
    # config_output_base_str = 'random_vd_'
    # config_list_file_name = 'random_vd_config_list.txt'

    # High efficiency VD
    # base_config_file = '../../configs/random_vd_higheff/base_random_vd_higheff.yaml'
    # config_output_dir = '../../configs/random_vd_higheff/'
    # config_output_base_str = 'random_vd_higheff_'
    # config_list_file_name = 'random_vd_higheff_config_list.txt'

    # High ADC VD
    base_config_file = '../../configs/random_vd_highadc/base_random_vd_highadc.yaml'
    config_output_dir = '../../configs/random_vd_highadc/'
    config_output_base_str = 'random_vd_highadc_'
    config_list_file_name = 'random_vd_highadc_config_list.txt'

    # Standard HD
    # The base config file to modify
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
        'max_cluster_time': {'mode': 'uniform', 'range': [0.05, 0.5]},
        'max_hit_time_diff': {'mode': 'uniform-constrained', 'range': [0.05, 0.5], 'constraint': 'max_cluster_time'},
        'max_hit_distance': {'mode': 'uniform', 'range': [100, 1100]},
        'min_hit_multiplicity': {'mode': 'grid', 'values': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        'min_neighbours': {'mode': 'grid', 'values': [1, 2]}
    }

    generate_random_configs(n_configs, starting_config_number, delete_old, base_config_file,
                            clustering_parameter_grid, config_output_dir, config_output_base_str, 
                            config_list_file_name=config_list_file_name)






