
from dataclasses import dataclass
from ase.db import connect
import numpy as np
from ase import units
from pprint import pprint
import matplotlib.pyplot as plt
from plot_params import get_plot_params
import matplotlib as mpl
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers
import string
import numpy as np
import csv
import matplotlib.image as mpimg
from collections import defaultdict

@dataclass
class IntensityRamanIR:
    modes: list # eigenmodes
    hnu: list # hnu in meV from frequency calculation
    dxi: float # finite difference field
    dz: float # finite difference spacing
    config: dict # configuration to be added to parser
    dbname: str # databasename
    atoms_list: list
    lambda_laser: float
    frequency_range: list
    direc: str
    
    def __post_init__(self):
        self.results = {}
        assert len(self.atoms_list) == 1; 'only one atoms object allowed'
        self.atoms = self.atoms_list[0]
        self.masses = self.atoms.get_masses()
        self.natoms = len([atom.index for atom in self.atoms if atom.symbol in ['C', 'O']])
        self.nu = 0.01 * units._e / units._c / units._hplanck * np.array(hnu) / 1000
        self.broadener = []

    def parsedb(self):
        for row in connect(self.dbname).select(**self.config):
            findiff = row.findiff
            direction = findiff[-2] ; index = int(findiff[:-2])
            _, field, _, dz, _ = row.displacement.split('_')            
            dz = float(dz) ; field = float(field)
            
            self.results.setdefault(index,{}).setdefault(direction,{}).setdefault(dz,{}).setdefault(field,{})['energy'] = row.energy
            self.results.setdefault(index,{}).setdefault(direction,{}).setdefault(dz,{}).setdefault(field,{})['dipole'] = row.dipole[-1]#row.dipole_field
            self.results.setdefault(index,{}).setdefault(direction,{}).setdefault(dz,{}).setdefault(field,{})['forces'] = row.forces

    def get_infrared_intensities(self):
        ## get the cartesian coordinates for 
        dmudR = np.zeros([3*self.natoms, 3]) # dipole change for dx in xyz for each index
        masses = np.zeros(self.natoms)
        indices = np.zeros(len(self.results))
        for i, index in enumerate(sorted(self.results)):
            dmudR_z = ( self.results[index]['m'][self.dz][0.0]['dipole'] \
                    -    self.results[index]['p'][self.dz][0.0]['dipole'] ) / 2 / self.dz
            dmudR[3*i,:] = np.array([0, 0, 0]) # no change in x for this system
            dmudR[3*i+1,:] = np.array([0, 0, 0]) # no change in y for this system
            dmudR[3*i+2,:] = np.array([0, 0, dmudR_z]) # assuming change only in z-dipole moment
            masses[i] = self.masses[index]
            indices[i] = int(index)

        dmudq = np.array( [ dmudR[j] / np.sqrt(masses[int(j/3)] * units._amu / units._me) \
            for j in range(3*self.natoms) ] )
        dmudQ = np.dot(dmudq.T, modes)
        dmudQ = dmudQ.T
        conv = (1.0 / units.Debye)**2 * units._amu / units._me
        intensities = np.array([sum(dmudQ[j]**2) for j in range(3*self.natoms)])
        self.intensities_IR = intensities * conv
        self.broaden_ir = []
        for i, omega in enumerate(self.nu):
            broaden = self._lorentzian(1, omega, self.frequency_range )
            self.broadener.append(broaden)
            self.broaden_ir.append(self.intensities_IR[i]*broaden)

    def _lorentzian(self, a, b1,energy):
        delta =  ( a / ( (energy - b1)**2 + a**2  ) )
        return delta

    def get_raman_cross_section(self):
        dalphadR = np.zeros([3*self.natoms, 3]) # raman tensor for each atom
        masses = np.zeros(self.natoms)
        indices = np.zeros(len(self.results))

        for i, index in enumerate(sorted(self.results)):
            # alpha_p =  self.results[index]['p'][self.dz][self.dxi]['energy'] \
            #          - 2*self.results[index]['p'][self.dz][0]['energy']\
            #          + self.results[index]['p'][self.dz][-self.dxi]['energy'] 
            # alpha_m =  self.results[index]['m'][self.dz][self.dxi]['energy'] \
            #          - 2*self.results[index]['m'][self.dz][0]['energy']\
            #          + self.results[index]['m'][self.dz][-self.dxi]['energy'] 
            # alpha_p =  -1 * self.results[index]['p'][self.dz][2*self.dxi]['energy']\
            #            + 16 * self.results[index]['p'][self.dz][self.dxi]['energy'] \
            #          - 30 * self.results[index]['p'][self.dz][0]['energy']\
            #            + 16 * self.results[index]['p'][self.dz][-self.dxi]['energy']\
            #          - self.results[index]['p'][self.dz][-2*self.dxi]['energy'] 

            # alpha_m =  -1 * self.results[index]['m'][self.dz][2*self.dxi]['energy']\
            #            + 16 * self.results[index]['m'][self.dz][self.dxi]['energy'] \
            #          - 30 * self.results[index]['m'][self.dz][0]['energy']\
            #            + 16 * self.results[index]['m'][self.dz][-self.dxi]['energy']\
            #          - self.results[index]['m'][self.dz][-2*self.dxi]['energy'] 
            d2Fdxi2 = self.results[index][self.direc][self.dz][self.dxi]['forces'][index] \
                    - 2 * self.results[index][self.direc][self.dz][0.0]['forces'][index] \
                    + self.results[index][self.direc][self.dz][-self.dxi]['forces'][index]
            d2Fdxi2 /= dxi**2
            masses[i] = self.masses[index]
            indices[i] = int(index)

            # alpha_p /= dxi**2
            # alpha_m /= dxi**2
            # dalphadR_zz = (alpha_p - alpha_m) / 2 / self.dz
            dalphadR[3*i,:] = np.array([0,0,0])
            dalphadR[3*i+1,:] = np.array([0,0,0])
            dalphadR[3*i+2,:] = np.array([0, 0, d2Fdxi2[-1]]) #np.array([0,0,dalphadR_zz])##

        dalphadq = np.array( [ dalphadR[j] / np.sqrt(masses[int(j/3)] * units._amu / units._me) \
            for j in range(3*self.natoms) ] ) * units._e / units._eps0 / 1e-10 # convert to AA**2/sqrt(amu)
        dalphadQ = np.dot(dalphadq.T, modes)
        dalphadQ = dalphadQ.T
        intensities = np.array([sum(dalphadQ[j]**2) for j in range(3*self.natoms)])
        self.raman_intensity = 45 * intensities # A^4/amu
        self.raman_cs = 5.8e-46 * (1e7 / self.lambda_laser - self.nu) / self.nu * self.raman_intensity \
                * (1 - np.exp(-self.nu / 201.56))**-1 
        self.broaden_cs = []
        for i, omega in enumerate(self.nu):
            broaden = self._lorentzian(1, omega, self.frequency_range )
            self.broaden_cs.append(self.raman_cs[i]*broaden)
        
            

def parsedb(dbname, results={}):
    for row in connect(dbname).select():
        charge = row.tot_charge
        sampling = row.sampling
        state = row.states 
        facet = row.facets

        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{}).setdefault(charge,{})['energy'] = row.energy
        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{}).setdefault(charge,{})['dipole'] = row.dipole[-1]
        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{}).setdefault(charge,{})['atoms'] = row.toatoms()
        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{}).setdefault(charge,{})['vibdata'] = row.data.vibdata
        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{}).setdefault(charge,{})['wf'] = row.wf


def vacuum_parsedb(dbname, results={}):
    for row in connect(dbname).select():
        sampling = row.sampling
        state = row.states 
        facet = row.facets

        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{})['energy'] = row.energy
        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{})['dipole'] = row.dipole[-1]
        results.setdefault(sampling,{}).setdefault(facet,{}).setdefault(state,{})['atoms'] = row.toatoms()

def boltzmann_coverages(xx, yy):
    ## return the boltzmann coverages for the xx components
    kBT = units.kB * 298.15 # K
    theta_co = np.exp(-1*xx / kBT) / ( 1 + np.exp(-1*yy / kBT) + np.exp(-1*xx / kBT) )
    return theta_co


if __name__ == '__main__':

    get_plot_params()
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.labelsize'] = 12
    # plt.rcParams['xtick.labelsize'] = 12
    # plt.rcParams['ytick.labelsize'] = 12

    ## databases to parse from
    charge_db = 'databases/charging_database.db'
    finitedf_db = 'databases/finite_difference.db'
    vacuum_db = 'databases/vacuum_database.db'

    results = {} # dict of results of charging calculation
    vac_results = {}
    parsedb(charge_db, results=results)
    vacuum_parsedb(vacuum_db, vac_results)
    frequency_range = np.linspace(1800,2100, 800)
    cmap = plt.get_cmap('coolwarm', 12)
    ls = ['-','--','.-']
    ls = {'CO_site_top_sp_implicit':'-' , \
        'CO_site_bridge_sp_implicit':'--', 'CO_site_fcc_sp_implicit':'-.'}
    metal_s = {'sampling_Pt':'v', 'sampling_Au':'o'}
    facet_s = {'facet_211':'v', 'facet_111':'o', 'facet_310':'*'}
    color_s = {'CO_site_bridge_sp_implicit':'tab:blue', \
        'CO_site_fcc_sp_implicit':'tab:green', 'CO_site_top_sp_implicit':'tab:orange'}

    # Figure for IR / Raman intensities
    figc = plt.figure(constrained_layout=True, figsize=(8.2,4.5))
    gs = figc.add_gridspec(2, len(results)+1)

    # Figure for water adsorption
    figw = plt.figure(constrained_layout=True, figsize=(8.2,3.5))
    gsw = figw.add_gridspec(1, 3)

    axc = []
    for i in range(len(results)):
        temp = []
        for j in [0,1]:
            temp.append(figc.add_subplot(gs[j,i:(i+1)]))
        axc.append(temp)
    axc = np.array(axc).T
    ## graph for the stark tuning rate and the dipole moments
    axmu = figc.add_subplot(gs[:,-1])
    axmu.set_xlabel(r'$ \mu_{\mathregular{CO}^*} - \mu_{*}$ / $e\AA$')
    axmu.set_ylabel(r'Stark Tuning Rate / $\mathregular{cm}^{-1} / \mathregular{V}$')


    ## graph for the boltzmann coverages of H2O and CO
    axb = figw.add_subplot(gsw[0,0])
    axb.set_xlabel(r'$\Delta G_{\mathregular{CO}}$ / eV')
    axb.set_ylabel(r'$\Delta G_{\mathregular{H}_2\mathregular{O}}$ / eV')

    axi = figw.add_subplot(gsw[0,1:])
    image = mpimg.imread('schematic/final_image.png')
    axi.imshow(image)
    axi.axis('off')


    ## add plot for gold
    axb.plot(0.0, 0.0, 'o', color='k')#jmol_colors[atomic_numbers['Au']], )
    axb.annotate('Au(310)', xy=(0.1,0.0), color='k')#jmol_colors[atomic_numbers['Au']], )
    axb.plot(-0.9, -0.1, 'o', color='k')#jmol_colors[atomic_numbers['Pt']], )
    axb.annotate('Pt(111)', xy=(-1.2,-0.4), color='k')#jmol_colors[atomic_numbers['Pt']], )
    axb.plot(-0.9, 0., 'o', color='k')#jmol_colors[atomic_numbers['Rh']],  )
    axb.annotate('Rh(111)', xy=(-1.3,0.3), color='k')#jmol_colors[atomic_numbers['Rh']], )
    axb.plot(-0.4, 0.2, 'o', color='k')#jmol_colors[atomic_numbers['Ni']],  )
    axb.annotate('Ni(111)', xy=(-0.4,0.4), color='k')#jmol_colors[atomic_numbers['Ni']], )
    

    ## plot the coverages based on boltzmann
    dG_CO = np.linspace(-1.5, 1.5)
    dG_H2O = np.linspace(-1.5, 1.5)
    xx, yy = np.meshgrid(dG_CO, dG_H2O)
    theta_co = boltzmann_coverages(xx, yy)
    cax = axb.contourf(xx, yy, theta_co, cmap=plt.cm.coolwarm)
    cbar = figc.colorbar(cax, ax=axb, ticks=[0.0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_ylabel(r'Boltzmann $\theta_{\mathregular{CO}}$ / ML')

    ## plot the raman / IR data
    for metal_ind, metal in enumerate(sorted(results)):
        # Store the potentials for the color plot
        # The intensities will be the scatter points
        all_IR_plot = defaultdict(list)
        all_Raman_plot = defaultdict(list)
        all_rhe_plot = defaultdict(list)

        figf, axf = plt.subplots(1, 1, figsize=(4,6), constrained_layout=True)

        for facet in sorted(results[metal]):

            fig, ax = plt.subplots(2, 1, figsize=(8,8), constrained_layout=True)

            for state_ind, state in enumerate(results[metal][facet]):
                if 'sp' not in state: continue
                if 'slab' in state: continue
                all_nu = []; all_pot = []
                for ic, charge in enumerate(sorted(results[metal][facet][state])):
                    # Get the IR and Raman intensities for each 
                    print('Metal %s, facet %s, state %s, charge %s'%(metal, facet, state, charge))
                    config = {'sampling':metal, 'states':state, 'tot_charge':charge, 'facets':facet, }
                    try:
                        modes = results[metal][facet][state][charge]['vibdata']['modes()']
                    except KeyError: 
                        print('Mode error', metal, facet, state, charge)
                        continue
                    hnu = results[metal][facet][state][charge]['vibdata']['frequency(meV)']
                    dxi = 0.1
                    dz = 0.01
                    direction = 'p'
                    atoms = results[metal][facet][state][charge]['atoms']
                    wf = results[metal][facet][state][charge]['wf'] 

                    ## feed into class
                    method = IntensityRamanIR(
                        modes=modes, hnu=hnu, dxi=dxi, dz=dz, config=config, dbname=finitedf_db,
                        atoms_list=[atoms], lambda_laser=1e2, frequency_range=frequency_range, direc=direction,
                    )
                    method.parsedb()
                    if len(method.results) == 0: 
                        print('Error parsing for results dict ')
                        continue
                    try:
                        method.get_infrared_intensities()
                        method.get_raman_cross_section()
                    except KeyError:
                        print('Error',facet, state)
                        continue
                    ir_intensities = method.broadener[-1] * results[metal][facet][state][charge]['vibdata']['ir_intensities((D/AA)^2 amu^-1)'][-1] 
                    rhe_potential = wf - 4.4 + 0.059 * 8.9

                    if rhe_potential < -2: 
                        print('Too negative')
                        continue
                    if 'sp' in state:
                        ax[0].plot(frequency_range,method.broaden_cs[-1]*1e41, ls[state], color=cmap(ic), label='%1.1f V'%rhe_potential)
                        ax[1].plot(frequency_range, ir_intensities, ls[state], color=cmap(ic))
                    
                    if 'sp' in state:
                        # Store everything based on the facet
                        all_IR_plot[facet].append(results[metal][facet][state][charge]['vibdata']['ir_intensities((D/AA)^2 amu^-1)'][-1] )
                        all_Raman_plot[facet].append(method.raman_intensity[-1])
                        all_rhe_plot[facet].append(rhe_potential)

                    all_nu.append(method.nu[-1])
                    all_pot.append(rhe_potential) 
                    if 'sp' in state:
                        axf.plot(rhe_potential, method.nu[-1], facet_s[facet], color=color_s[state])
                        axc[0,metal_ind].plot(rhe_potential, method.nu[-1], facet_s[facet], color=color_s[state], alpha=0.5)

                try: 
                    fit = {}
                    fit['fit'] = np.polyfit(all_pot, all_nu, 1)
                    fit['p'] = np.poly1d(fit['fit'])
                    axc[0,metal_ind].plot(all_pot, fit['p'](all_pot), '-', alpha=0.25, color=color_s[state])
                    xlabel = '%s(%s)'%(metal.replace('sampling_',''), facet.replace('facet_',''))
                    vac_state = state.replace('_sp_implicit','').replace('_implicit','')
                    dmu = vac_results[metal][facet][vac_state]['dipole'] - vac_results[metal][facet]['state_slab']['dipole'] 

                    if 'top' in state:
                        marker = 'v'
                    elif 'bridge' in state:
                        marker = 'o'
                    else:
                        marker = '*'
                    axmu.plot(dmu, fit['fit'][0], marker, color=jmol_colors[atomic_numbers[metal.replace('sampling_','')]])
                except TypeError:
                    continue

            ax[0].set_ylabel(r'Raman Cross Section x$10^{41}$', )
            ax[1].set_xlabel(r'Frequency / cm$^{-1}$', )
            ax[1].set_ylabel(r'IR Intensity', )
            fig.savefig('output/%s_%s_ir_raman_compare.pdf'%(metal,facet))
        
        axf.legend(loc='best', frameon=False, )
        # axf.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        axf.set_ylabel(r'$\nu \left ( \mathrm{cm}^{-1} \right )$')
        axf.set_xlabel(r'Potential vs. RHE')
        figf.savefig('output/%s_frequencies_compare.pdf'%metal)

        axmu.annotate(metal.replace('sampling_',''), xy=(0.8, 0.2+0.1*metal_ind), \
                        xycoords = 'axes fraction',\
                        color=jmol_colors[atomic_numbers[metal.replace('sampling_','')]])

        for i, facet in enumerate(all_IR_plot):
            ma = axc[1,metal_ind].scatter(all_Raman_plot[facet], all_IR_plot[facet], 
                                          c=all_rhe_plot[facet], marker=facet_s[facet], cmap='coolwarm')
            if i == 0:
                ma.set_clim([-2,2])
        if metal_ind == len(results)-1:
            cb = fig.colorbar(ma, ax=axc[1,metal_ind])
            cb.set_label('Potential vs. RHE')
        if metal_ind ==0:
            axc[0,metal_ind].set_ylabel(r'$\nu$ / $\mathregular{cm}^{-1}$')
            axc[1,metal_ind].set_ylabel(r'$\mathrm{I}_{\mathrm{IR}}$ / $(D/\AA)^2 \mathregular{amu}^{-1}$')
        axc[1,metal_ind].set_xlabel(r'$\mathrm{I}_{\mathrm{Raman}}$ / $\AA^4/\mathregular{amu}$')
        axc[0,metal_ind].set_xlabel(r'Potential / V vs. RHE')
        # axc[0,metal_ind].set_title(metal.replace('sampling_',''), )
        axc[0,metal_ind].annotate(metal.replace('sampling_', ''), xy=(0.1, 0.5), xycoords='axes fraction')
        axc[0,metal_ind].set_ylim([1700, 2100])
        axc[0,metal_ind].tick_params(direction='out')
        axc[1,metal_ind].tick_params(direction='out')
        if metal_ind != 0:
            axc[0,metal_ind].annotate('top', color='tab:orange', xy=(0.1,0.9), xycoords='axes fraction')
            axc[0,metal_ind].annotate('bridge', color='tab:blue', xy=(0.4,0.9), xycoords='axes fraction')
            axc[0,metal_ind].annotate('fcc', color='tab:green', xy=(0.8,0.9), xycoords='axes fraction')
        else:
            axc[0,metal_ind].annotate('top', color='tab:orange', xy=(0.7,0.6), xycoords='axes fraction')
            for facet, marker in facet_s.items():
                axc[0,metal_ind].plot([],[], marker, color='k',) #label=facet.replace('facet_',''))
            axc[0,metal_ind].legend(loc='best', frameon=False, )

    for a in [axmu]:
        a.plot([], [], 'v', label='top', color='k')
        a.plot([], [], 'o', label='bridge', color='k')
        a.plot([], [], '*', label='fcc', color='k')
        # a.legend(loc='best', frameon=False, )
        # a.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        a.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    for facet, marker in facet_s.items():
        axc[0,0].plot([],[], marker, color='k', label=facet.replace('facet_',''))
        # axc[0,0].legend(loc='best', frameon=False, )
        axc[0,0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)

    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(list(axc.flatten()[0:4]) + [axmu] + list(axc.flatten()[4:]) ):
        a.annotate(alphabet[i]+')', xy=(-0.15, 1.1), xycoords='axes fraction', )

    for i, a in enumerate([axb, axi]):
        a.annotate(alphabet[i]+')', xy=(-0.1, 1.1), xycoords='axes fraction', )

    figc.savefig('output/IR_Raman_overall_compare.pdf')
    figw.savefig('output/IR_Raman_water_adsorption.pdf')
    figc.savefig('output/IR_Raman_overall_compare.png', dpi=1200)
    figw.savefig('output/IR_Raman_water_adsorption.png', dpi=1200)

    # Store the xy data for each axis in a csv format
    with open('output/figure_4.csv', 'w') as f:
        writer = csv.writer(f)
        for i, a in enumerate(axc.flatten().tolist()[0:4] + [axmu]):
            # write a blank row
            writer.writerow([])
            # Write the index of the plot
            writer.writerow([alphabet[i]])
            for line in a.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                if len(x_data)==1:
                    x_data = x_data[0]
                    y_data = y_data[0]
                    writer.writerow([x_data, y_data])

        for j, a in enumerate(axc.flatten().tolist()[4:]):
            writer.writerow([])
            data = a.collections[0].get_offsets()
            x_data, y_data = np.array(data).T
            writer.writerow(alphabet[j+i+1])
            for k in range(len(x_data)):
                writer.writerow([x_data[k], y_data[k]])

    with open('output/figure_5.csv', 'w') as f:
        writer = csv.writer(f)
        for i, a in enumerate([axb]):
            writer.writerow([])
            for line in a.lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                writer.writerow([x_data, y_data])



