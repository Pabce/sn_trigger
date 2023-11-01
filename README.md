# Supernova trigger PDS algorithm
### Word of caution: the hit data is at this point unreliable so do not trust any results you get from this code.

This package computes the supernova trigger efficiency for the DUNE FD-VD, using the PDS. It takes `.root` files containing the optical hits for supernova events and the radiological backgrounds and outputs the efficiency at a given distance, optimized for a range of clustering parameters. The complete efficiency curve (all distances) is also computable.

## Download
Just clone the repository into a local directory.

    git clone https://github.com/Pabce/sn_trigger

## Configure
The `parameters.py` file contains a list of parameters used to run the algorithm. First thing you will want to specify are the locations of the SN signal and background files. For example, if running on EOS you can access:
```python
EVENT_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/prod_snnue_pds/"
BG_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/prod_background_pds/"
```
(Ask me for access if you don't have it.)

Other than that, the parameters you most likely will want to change are the average energy and pinching parameter $\alpha$ of the SNB spectrum. You can check `plotter_basico.ipynb` to see how the original and interacted (flux $\cdot$ cross-section) spectra look like for different sets of the parameters. For example, the GKVM model interacted spectrum is very well approximated by:
```python
AVERAGE_ENERGY = 23.0 # MeV
ALPHA = 5.0 # Dimensionless 
```
You may also want to play around the clustering parameters over which to optimize.

## Usage
Once you have set the `parameters.py` file to your liking, you can simply run:
```bash
python hit_stat.py
```
This will run the algorithm for the set parameters and save an output file with the optimized clustering parameters, trained BDT, the efficiency data and the full list of set parameters. If you now want to generate the efficiency curve you can run:
```bash
python hit_stat.py --eff-curve
```
This will load the optimized clustering parameters and BDT computed with the previous command and calculate the efficiency for a range of distances (if no file is available, it will run the whole computation).
You can plot these efficiency curves in `plotter_basico.ipynb` by specifying the parameters with which you run the simulation.

You can override certain parameters directly from the command line, like the output file name. For more information you can run

```
python hit_stat.py --help
```

```
usage: hit_stat.py [-h] [-e ENERGY] [-a ALPHA] [-m MODE] [-o OUTPUT] [-d DTO] [--eff-curve] [-i INPUT] [--eff-curve-output EFF_CURVE_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -e ENERGY, --energy ENERGY
                        Average energy of the SN neutrino spectrum
  -a ALPHA, --alpha ALPHA
                        Pinching parameter of the SN neutrino spectrum
  -m MODE, --mode MODE  Simulation mode ('aronly' or 'xe')
  -o OUTPUT, --output OUTPUT
                        Name of the output file for the efficiency data. If not specified, the name will be generated randomly.
  -d DTO, --dto DTO     Distance to optimize for (in kpc)
  --eff-curve           Attemp to compute the efficiency curve vs. distance for the current parameters, if the efficiency data file exists. Else, we will
                        run the whole algorithm first.
  -i INPUT, --input INPUT
                        Name of the input file for the efficiency data, when computing the efficiency curve. If not specified, the file matching the
                        current parameters will be used.
  --eff-curve-output EFF_CURVE_OUTPUT
                        Name of the output file for the efficiency curve. If not specified, the name will be generated randomly.
```