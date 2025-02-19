import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file")
    parser.add_argument("-i", "--input", type=str, help="Path to the input file for the intial stage of the algorithm")
    parser.add_argument("-o", "--output", type=str, help="Path to the output data file for the final stage of the algorithm")
    parser.add_argument("--info_output", type=str, help="Path to the output file for the run information")

    args = parser.parse_args()

    # Raise an error if no configuration file is provided
    if not args.config:
        parser.error("No configuration file provided")
    
    return args.config, args.input, args.output, args.info_output