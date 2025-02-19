import numpy as np
import logging

from gui import console
import data_loader as dl

# TODO: Add separate spectra for CC and ES interactions
# TODO: Add explicit support for non-pinched models
# TODO: Add support for non-default time profiles (specified in config file)

MODEL_PINCHED_PARAMETERS = {
    "Livermore": {
        'alpha': 2.8, 
        'average_energy': 14.4,
        'interaction_number_10kpc': 2684
        },
    "Garching": {
        'alpha': 4.5,
        'average_energy': 12.2,
        'interaction_number_10kpc': 882
        },
    "GKVM": {
        'alpha': 5.0,
        'average_energy': 23.0,
        'interaction_number_10kpc': 3295
        }
}

class SupernovaSpectrum:

    def __init__(self, spectrum_values, energy_bins, interaction_number_10kpc=3000, time_profile=None, parameters={}):
        # If spectrum is not normalised to 1 (within threshold), warn the user and normalise it 
        eps = 1e-6
        if np.abs(np.sum(spectrum_values) - 1) > eps:
            logging.warning(f"Warning: Supernova spectrum is not normalised to 1 (sums to {np.sum(spectrum_values)}) Normalising it now.")
            spectrum_values = spectrum_values / np.sum(spectrum_values)

        self.spectrum_values = spectrum_values
        self.energy_bins = energy_bins

        # Values of v_e-CC interactions at 10 kpc for a 40 kton LArTPC
        self.interaction_number_10kpc = interaction_number_10kpc

        # Set the time profile (only default is supported at the moment)
        #self.set_time_profile(time_profile)
        if time_profile:
            self.time_profile_x, self.time_profile_y = time_profile

        # If we have any parameters (e.g. by calling the from_model_name method),
        # they will be used for the __str__ method
        self.parameters = parameters
    
    def __str__(self) -> str:
        if self.parameters:
            if 'model_name' in self.parameters.keys():
                return f"{self.parameters['model_name']}"
            else:
                return f"Model-with-parameters:{self.parameters}"
        else:
            # TODO: Add support for non-pinched models
            return "Custom_model"
        

    @classmethod
    def from_model_name(cls, model_name, time_profile=None):
        if model_name not in MODEL_PINCHED_PARAMETERS.keys():
            raise ValueError(f"Model name {model_name} not found in the list of available models.")
        
        return cls.from_pinched_spectrum(**MODEL_PINCHED_PARAMETERS[model_name], time_profile=time_profile, parameters={'model_name': model_name})
    
    @classmethod
    def from_pinched_spectrum(cls, average_energy=20.0, alpha=2.0, energy_bins=None, interaction_number_10kpc=3000, time_profile=None, parameters={}):
        
        if energy_bins is None:
            energy_bins = np.linspace(4, 70, 100)
        
        spectrum_values, energy_bins = cls.pinched_spectrum_histogram(energy_bins, average_energy, alpha)
        parameters.update({'average_energy': average_energy, 'alpha': alpha})

        return cls(spectrum_values, energy_bins, interaction_number_10kpc, time_profile=time_profile, parameters=parameters)

    # def set_time_profile(self, time_profile):
    #     if time_profile == 'default':
    #         # Load the time profile
    #         data_loader = dl.DataLoader(self.config, logging_level=logging.INFO)
    #         time_profile_x, time_profile_y = data_loader.load_time_profile()

    #         self.time_profile_x = time_profile_x
    #         self.time_profile_y = time_profile_y
    #     else:
    #         raise NotImplementedError("Only the default time profile is supported at the moment.")
            
    
    @staticmethod
    def pinched_spectrum_histogram(energy_bins, average_energy, alpha):
        # Compute the spectrum values for the central values of the energy bins
        central_energies = (energy_bins[1:] + energy_bins[:-1]) / 2
        spectrum_values = pinched_spectrum(central_energies, average_energy, alpha)

        # Normalize the spectrum values to 1
        spectrum_values /= np.sum(spectrum_values)

        return spectrum_values, energy_bins

    def event_number_to_distance(self, event_number, tpc_size):
        return event_number_to_distance(event_number, self.interaction_number_10kpc, tpc_size=tpc_size)

    def distance_to_event_number(self, distance, tpc_size):
        return distance_to_event_number(distance, self.interaction_number_10kpc, tpc_size=tpc_size)


def event_number_to_distance(event_number, base_event_number_40kt, tpc_size=10):
    return np.sqrt(base_event_number_40kt * 10**2 * (tpc_size/40) * 1/event_number)

def distance_to_event_number(distance, base_event_number_40kt, tpc_size=10):
    return base_event_number_40kt * 10**2 * (tpc_size/40) * 1/distance**2


def pinched_spectrum(energy, average_energy, alpha):
    # Compute the spectrum value for the given energy
    return (energy / average_energy) ** alpha * np.exp(- (alpha + 1) * energy/average_energy)


def sample_indices_by_energy(spectrum_values, energy_bins, sn_energies, n_samples):
    # Energy histogram should be normalized to 1
    cdf = np.cumsum(spectrum_values)
    cdf /= cdf[-1]
    # Append a 0 to the cdf at the left
    cdf = np.insert(cdf, 0, 0.)

    rng = np.random.rand(n_samples)
    cdf_indices = np.searchsorted(cdf, rng)

    event_indices = []
    for i, cdf_index in enumerate(cdf_indices):
        left_energy = energy_bins[cdf_index - 1]
        right_energy = energy_bins[cdf_index]

        # Get event indices with energy between left and right energy
        #indices = [i for i, e in enumerate(sn_energies) if e >= left_energy and e < right_energy]
        indices = np.where((sn_energies >= left_energy) & (sn_energies < right_energy))[0]

        if len(indices) == 0:
            console.log(cdf[0])
            console.log(cdf_indices[i], rng[i])
            raise ValueError(f"No indices found for the given energy range. "
                             + f"Left: {left_energy}, Right: {right_energy}. "
                             + f"Try loading more SN events or making the energy histogram bins wider.")

        index = np.random.choice(indices)
        event_indices.append(index)

    return np.array(event_indices)

# Function to weight a bunch of events by the spectrum energies
def weigh_by_supernova_spectrum(sn_energies, supernova_spectrum):
    weights = []
    for i, _ in enumerate(sn_energies):
        energy = sn_energies[i]
        # Find the energy bin corresponding to the energy
        energy_bin_index = np.searchsorted(supernova_spectrum.energy_bins, energy)
        # Add the weight
        weight = supernova_spectrum.spectrum_values[energy_bin_index-1]
        weights.append(weight)
    
    weights = np.array(weights)
    weights /= np.sum(weights)
    return weights