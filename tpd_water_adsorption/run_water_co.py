
import numpy as np
import matplotlib.pyplot as plt
from tpd_analyse import tpd
from glob import glob
import os.path as op
import json
from ase.io import read
from ase.build import molecule
from ase.thermochemistry import IdealGasThermo
from pprint import pprint
from plot_params import get_plot_params_Arial
get_plot_params_Arial()
from pathlib import Path
import os

if __name__ == '__main__':

    metals = ['Ni']
    facets = ['111']
    species = [ 'CO', 'H2O']

    # create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    for metal in metals:
        for facet in facets:
            for species_ in species:
                all_E0 = []
                fig, ax = plt.subplots(1, 3, figsize=(12,4), squeeze=False)
                # Read in the data
                files = glob(op.join(f'{metal}_{facet}', f'{species_}', '*.csv' ))
                # get the inputs from the json files stored in each folder
                with open(op.join(f'{metal}_{facet}', f'{species_}', 'inputs.json')) as f:
                    inputs = json.load(f)
                small_ads = molecule(species_)
                # get the vibration data 
                with open(op.join(f'molecules', f'{species_}', 'vibrations.json')) as f:
                    vib = json.load(f)
                # create the ASE class for the adsorbate
                Thermo = IdealGasThermo(vib_energies=vib["gas_phase"],
                                        potentialenergy=0,
                                        geometry='linear',
                                        atoms=small_ads)
                TPD = tpd.PlotTPD(
                            exp_data=files,
                            thermo_gas=Thermo,
                            **inputs,
                            )
                TPD.get_results()
                results_all = TPD.results

                for i in results_all:
                    for exposure, results in results_all[i].items():
                        # plot the results for a quick check
                        ax[0,0].plot(results['temperature'], results['normalized_rate'], '-o', label=exposure)
                        ax[0,1].plot(results['theta_rel'], results['Ed'], 'o')
                        ax[0,1].plot(results['theta_rel'], results['Ed_fitted'], '-')
                        ax[0,2].axhline(y=results['E0'], ls='--')                        
                        ax[0,2].plot(results['theta_rel'], results['configurational_entropy'], '-')
                        ax[0,2].plot(results['theta_rel'], results['ads_ads_interaction'], '-')
                        all_E0.append(results['E0'])

                mean_E0 = np.mean(all_E0)
                std_E0 = np.std(all_E0)
                print(all_E0)
                print(f'Delta E0 for %s: %1.2f eV with std dev %1.4f'%(species_, mean_E0, std_E0)) 

                ax[0,0].set_xlabel('Temperature (K)')
                ax[0,0].set_ylabel('Normalised rate')
                ax[0,0].legend(loc='best', fontsize=10)
                ax[0,1].set_xlabel(r'$\theta_{rel}$ ')
                ax[0,1].set_ylabel('$E_d$ (eV)')
                ax[0,2].set_ylabel(r'Contributions to $E_d$')
                ax[0,2].set_xlabel(r'$\theta_{rel}$ ')
                fig.suptitle(f'{metal} ({facet}) {species_}')
                fig.tight_layout()
                fig.savefig(op.join(output_dir , f'{metal}_{facet}_{species_}.png'))









