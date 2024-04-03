import scipy.constants as SI
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
import copy
import functools
import matplotlib

SI.r_e = SI.physical_constants['classical electron radius'][0]

class ColliderCostModel():
    
    # constructor
    def __init__(self):
        
        ## NAMESPACES
        
        self.bunch_pattern = SimpleNamespace()
        self.pwfa = SimpleNamespace()
        self.driver_linac = SimpleNamespace()
        
        self.linac1 = SimpleNamespace()
        self.linac1.beam = SimpleNamespace()
        self.linac1.pwfa_linac = SimpleNamespace()
        self.linac1.pwfa_linac.stage = SimpleNamespace()
        self.linac1.pwfa_linac.interstage = SimpleNamespace()
        self.linac1.pwfa_linac.driver = SimpleNamespace()
        self.linac1.pwfa_linac.driver_linac = SimpleNamespace()
        self.linac1.pwfa_linac.kicker_tree = SimpleNamespace()
        self.linac1.rf_linac = SimpleNamespace()
        self.linac1.injector = SimpleNamespace()
        self.linac1.bds = SimpleNamespace()
        self.linac1.source = SimpleNamespace()
        self.linac1.damping_ring = SimpleNamespace()
        
        self.linac2 = SimpleNamespace()
        self.linac2.beam = SimpleNamespace()
        self.linac2.pwfa_linac = SimpleNamespace()
        self.linac2.pwfa_linac.stage = SimpleNamespace()
        self.linac2.pwfa_linac.interstage = SimpleNamespace()
        self.linac2.pwfa_linac.driver = SimpleNamespace()
        self.linac2.pwfa_linac.driver_linac = SimpleNamespace()
        self.linac2.pwfa_linac.kicker_tree = SimpleNamespace()
        self.linac2.rf_linac = SimpleNamespace()
        self.linac2.injector = SimpleNamespace()
        self.linac2.bds = SimpleNamespace()
        self.linac2.source = SimpleNamespace()
        self.linac2.damping_ring = SimpleNamespace()
        

        ## FREE PARAMETERS
        
        # free parameters
        self.energy_centerofmass = 250e9 # [eV]
        self.energy_asymmetry = 1
        self.linac1.beam.num_particles = 2e10 # [electrons]
        self.linac2.beam.num_particles = 2e10 # [positrons]
        self.linac1.beam.energy_initial = 3e9 # [eV]
        self.linac2.beam.energy_initial = 3e9 # [eV]
        self.linac1.beam.charge_sign = -1 # [electrons]
        self.linac2.beam.charge_sign = +1 # [positrons]
        self.pwfa.accel_gradient = 2e9 # [V/m]
        self.pwfa.transformer_ratio = 1.5
        self.pwfa.num_stages = 32
        self.pwfa.extraction_efficiency = 0.4
        self.pwfa.depletion_efficiency = 0.75
        self.emittance_asymmetry = 2
        self.norm_emittance_x_min = 10e-6 # [m rad]
        self.norm_emittance_y_min = 35e-9 # [m rad]
        self.betaxIP_min = 3.3e-3 # [m]
        self.betayIP_min = 0.1e-3 # [m]
        
        self.BDR_max = 1e-6
        self.eta_prod_positrons = 0.5

        self.bunch_pattern.bunch_spacing = 10e-9 # [s]
        self.bunch_pattern.bunches_per_train = 100
        self.bunch_pattern.reprate_trains = 100 # [Hz]

        self.driver_linac.accel_gradient = 15e6 # [V/m]
        self.driver_linac.efficiency_wallplug_to_rf = 0.55
        self.driver_linac.cavity_frequency = 1e9 # [Hz]
        self.driver_linac.cell_length = 2.3 # [m]
        self.driver_linac.peak_power_klystrons = 50e6 # [W]
        
        # interstage scaling
        self.pwfa.interstage_length_scaling = 4/np.sqrt(10e9) # [m/sqrt(eV)]
        self.pwfa.driver_jitter_transverse = 100e-9 # [m]
        self.pwfa.kickers_per_side = 4

        # sources and damping rings
        self.source_emittance_electron = 2e-7 # [m rad]
        self.source_emittance_positron = 1e-2 # [m rad]
        self.max_damping_times_per_ring = 7
        
        # choices
        self.linac1.pwfa_linac.enable = True
        self.linac2.pwfa_linac.enable = False
        self.combined_driver_positron_linac = False
        self.use_driver_turnaround = False
        self.use_coolcopper_rf = False
        self.use_superconducting_rf = False
        self.combined_driver_tunnel = False
        self.combined_damping_rings = False
        #self.shared_driver_linac = False

        # RF technology specific efficiencies
        self.scrf_efficiency_wallplug_to_rf = 0.65
        self.ccrf_efficiency_wallplug_to_rf = 0.55
        self.ncrf_efficiency_wallplug_to_rf = 0.55

        # cost length scalings
        self.cost = SimpleNamespace()
        self.cost.per_length = SimpleNamespace()
        self.cost.per_length.tunnel = 0.087e6 # [ILCU/m]
        self.cost.per_length.rf_linac = 0.54e6 # [ILCU/m]
        self.cost.per_length.plasma_stage = 0.54e6 # [ILCU/m]
        self.cost.per_length.plasma_interstage = 0.3e6 # [ILCU/m]
        self.cost.per_length.plasma_kicker_beamline = 0.2e6 # [ILCU/m]
        self.cost.per_length.turn_arounds = 0.2e6 # [ILCU/m]
        self.cost.per_length.BDS = 0.05e6 # [ILCU/m]
        self.cost.per_length.damping_ring = 0.3e6 # [ILCU/m]
        
        # cost power scalings
        self.cost_per_power = 0.2/(3600*1000)# ILCU/J (0.25 ILCU per kWh)
        self.cost_per_power_infrastructure_linacs = 1.12 # [ILCU/W]
        self.cost_per_power_infrastructure_other = 0.78 # [ILCU/W]
        self.cost_per_beam_power_dumps = 67e6/10.5e6 # [ILCU/W]

        # cost time scalings
        self.maintenance_labor_per_construction_cost = 100/1e9 # [FTE/BILCU/year] # people required for maintaining the machine
        self.cost_per_labor = 0.07e6 # [ILCU/FTE] 
        self.uptime_percentage = 0.6 # TODO: how does this scale with machine complexity (number of critical components)

        # constant costs
        self.cost_per_source = 9.4e6 # [ILCU/source]
        self.cost_per_IP = 50e6 # [ILCU]

        # construction overheads
        self.overheads = SimpleNamespace()
        self.overheads.construction = SimpleNamespace()
        self.overheads.construction.design_and_development = 0.1
        self.overheads.construction.controls_and_cabling = 0.15
        self.overheads.construction.installation_and_commissioning = 0.15
        self.overheads.construction.management_inspection = 0.12

        # power overheads
        self.overheads.power = SimpleNamespace()
        self.overheads.power.total = 0.25
        
        #self.integrated_lumi_required = 1e46 # one inverse attobarn [for Higgs @ 250 GeV]
        self.integrated_lumi_required_minimum = 0.8e46 # one inverse attobarn [for Higgs @ 250 GeV]
        self.integrated_lumi_required_per_com_energy = 1.67e34 # one inverse attobarn (scales as 1/s) [per m^2/eV^2]
        self.integrated_lumi_required_gamma_gamma_500GeV = 1e46 # TODO: one inverse attobarn  (scales as log s)
        self.integrated_lumi_required_gamma_gamma_model = lambda E: self.integrated_lumi_required_gamma_gamma_500GeV/(0.0258*(4.5942*np.log10(E/1e9)-6.1765)**2)
        
        # emissions
        self.emissions_per_tunnel_length = 20 # [ton CO2e/m] from 20 kton/km
        self.emissions_per_energy_usage = 15/(1e9*3600) # [ton CO2e/J] from 20 ton/GWh
        self.carbon_tax_per_emissions = 140 # [ILCU per ton CO2e] using IEA 2015 estimate

        ## PERFORM CALCULATION
        
        # initial calculation
        self.calculate_all()


    # HELPER FUNCTIONS

    # set parameters
    def set_parameters(self, params, recalculate=True):
        for key in params.keys():
            self.set(key, params[key])
        if recalculate:
            self.calculate_all()

    
    # helper functions for setting and getting attributes
    def set(self, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.get(pre) if pre else self, post, val)

    def get(self, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [self] + attr.split('.'))
    

    ## PREDEFINED PARAMETER SETS

    def set_as_HALHF(self, energy_centerofmass=250e9, use_coolcopper_rf=False):
        self.energy_asymmetry = 4
        self.emittance_asymmetry = 2
        self.energy_centerofmass = energy_centerofmass
        self.linac1.pwfa_linac.enable = True
        self.linac2.pwfa_linac.enable = False
        self.linac1.beam.charge_sign = -1 # [electrons]
        self.linac2.beam.charge_sign = +1 # [positrons]
        self.linac1.beam.num_particles = 1e10 # [electrons]
        self.linac2.beam.num_particles = 3e10 # [positrons]
        self.linac1.beam.energy_initial = 5e9 # [eV]
        self.linac2.beam.energy_initial = 2.86e9 # [eV]
        self.use_coolcopper_rf = use_coolcopper_rf
        self.pwfa.num_stages = 16
        self.pwfa.accel_gradient = 2e9 # [V/m]
        self.pwfa.transformer_ratio = 1.5
        self.pwfa.extraction_efficiency = 0.5
        self.pwfa.depletion_efficiency = 0.8
        self.driver_linac.accel_gradient = 25e6 # [V/m]
        self.bunch_pattern.bunch_spacing = 80e-9 # [s]
        self.bunch_pattern.bunches_per_train = 100
        self.bunch_pattern.reprate_trains = 100 # [Hz]
        self.calculate_all()

    def set_as_original_HALHF(self, energy_centerofmass=250e9):
        self.energy_asymmetry = 4
        self.emittance_asymmetry = 2
        self.energy_centerofmass = energy_centerofmass
        self.linac1.pwfa_linac.enable = True
        self.linac2.pwfa_linac.enable = False
        self.combined_driver_positron_linac = True
        self.use_driver_turnaround = True
        self.pwfa.kickers_per_side = 16
        self.linac1.beam.charge_sign = -1 # [electrons]
        self.linac2.beam.charge_sign = +1 # [positrons]
        self.linac1.beam.num_particles = 1e10 # [electrons]
        self.linac2.beam.num_particles = 4e10 # [positrons]
        self.linac1.beam.energy_initial = 5e9 # [eV]
        self.linac2.beam.energy_initial = 2.86e9 # [eV]
        self.pwfa.num_stages = 16
        self.pwfa.accel_gradient = 6.4e9 # [V/m]
        self.pwfa.transformer_ratio = 1
        self.pwfa.extraction_efficiency = 0.5
        self.pwfa.depletion_efficiency = 0.75
        self.driver_linac.accel_gradient = 25e6 # [V/m]
        self.bunch_pattern.bunch_spacing = 80e-9 # [s]
        self.bunch_pattern.bunches_per_train = 100
        self.bunch_pattern.reprate_trains = 100 # [Hz]
        self.calculate_all()

    def set_as_pwfa_gammagamma(self, energy_centerofmass=10e12):
        self.energy_asymmetry = 1
        self.energy_centerofmass = energy_centerofmass
        self.linac1.pwfa_linac.enable = True
        self.linac2.pwfa_linac.enable = True
        self.linac1.beam.charge_sign = -1 # [electrons]
        self.linac2.beam.charge_sign = -1 # [positrons]
        self.norm_emittance_x_min = 0.5e-6 # [m rad]
        self.norm_emittance_y_min = 0.5e-6 # [m rad]
        self.calculate_all()

    def set_as_ILC(self, energy_centerofmass=250e9): # TODO: implement superconducting RF
        self.energy_asymmetry = 1
        self.energy_centerofmass = energy_centerofmass
        self.linac1.pwfa_linac.enable = False
        self.linac2.pwfa_linac.enable = False
        self.use_superconducting_rf = True
        self.use_coolcopper_rf = False
        self.combined_damping_rings = True
        self.bunch_pattern.bunch_spacing = 554e-9 # [s]
        self.bunch_pattern.bunches_per_train = 1312
        self.bunch_pattern.reprate_trains = 5 # [Hz]
        self.linac1.beam.num_particles = 2e10 # [electrons]
        self.linac2.beam.num_particles = 2e10 # [positrons]
        self.calculate_all()
        
    def set_as_C3(self, energy_centerofmass=250e9):
        self.use_coolcopper_rf = True
        self.use_superconducting_rf = False
        self.linac1.pwfa_linac.enable = False
        self.linac2.pwfa_linac.enable = False
        self.energy_asymmetry = 1
        self.energy_centerofmass = energy_centerofmass
        self.linac1.beam.num_particles = 1e-9/SI.e # [electrons]
        self.linac2.beam.num_particles = 1e-9/SI.e # [positrons]
        self.bunch_pattern.reprate_trains = 120 # [Hz]
        if energy_centerofmass == 250e9:
            self.bunch_pattern.bunch_spacing = 5.26e-9 # [s]
            self.bunch_pattern.bunches_per_train = 133
        elif energy_centerofmass >= 500e9:
            self.bunch_pattern.bunch_spacing = 3.55e-9 # [s]
            self.bunch_pattern.bunches_per_train = 75
        self.calculate_all()
        
    
    def calculate_all(self):

        # TODO:
        # incorporate limitations on charge (positrons and electrons; damping ring power, RF transverse wakefields, field beam loading)
        # incorporate problems due to high driver current (CSR, need for compressor)
        # incorporate power requirement in the damping ring (limits the positron charge, sets length of RF and wigglers)
        # incorporate cost per klystron
        # incorporate superconducting linacs (limit on peak field, cost of cooling etc.)
        # luminosity scaling per energy
        
        # final beam energies
        self.linac1.beam.energy_final = (self.energy_centerofmass/2) * self.energy_asymmetry
        self.linac2.beam.energy_final = (self.energy_centerofmass/2) / self.energy_asymmetry
        
        # emittances
        self.linac1.beam.norm_emittance_x = self.norm_emittance_x_min * max(1, self.energy_asymmetry**self.emittance_asymmetry)
        self.linac1.beam.norm_emittance_y = self.norm_emittance_y_min * max(1, self.energy_asymmetry**self.emittance_asymmetry)
        self.linac2.beam.norm_emittance_x = self.norm_emittance_x_min * max(1, self.energy_asymmetry**(-self.emittance_asymmetry))
        self.linac2.beam.norm_emittance_y = self.norm_emittance_y_min * max(1, self.energy_asymmetry**(-self.emittance_asymmetry))

        # gammas
        self.linac1.beam.gamma_initial = self.linac1.beam.energy_initial/(SI.m_e*SI.c**2/SI.e)
        self.linac2.beam.gamma_initial = self.linac2.beam.energy_initial/(SI.m_e*SI.c**2/SI.e)
        self.linac1.beam.gamma_final = self.linac1.beam.energy_final/(SI.m_e*SI.c**2/SI.e)
        self.linac2.beam.gamma_final = self.linac2.beam.energy_final/(SI.m_e*SI.c**2/SI.e)
        
        # charges
        self.linac1.beam.charge = SI.e * self.linac1.beam.num_particles
        self.linac2.beam.charge = SI.e * self.linac2.beam.num_particles
        

        ## SOURCES AND DAMPING RINGS
        
        # go through linacs separately
        for linac in [self.linac1, self.linac2]:

            ## arm description
            if linac.pwfa_linac.enable:
                linac.description = 'PWFA'
            else:
                if self.use_coolcopper_rf:
                    linac.description = 'COOL COPPER RF'
                elif self.use_superconducting_rf:
                    linac.description = 'SUPERCONDUCTING RF'
                else:
                    linac.description = 'RF'
            if linac.beam.charge_sign < 0:
                linac.description = linac.description + ' ELECTRON ARM'
            else:
                linac.description = linac.description + ' POSITRON ARM'
            
            ## SOURCE
            
            # positron source
            if linac.beam.charge_sign > 0:
                linac.source.norm_emittance_initial = self.source_emittance_positron
            else:
                linac.source.norm_emittance_initial = self.source_emittance_electron

            
            ## DAMPING RINGS
            
            # positron damping rings (all assumed to be in the same tunnel)
            
            linac.damping_ring.damping_ratio = linac.source.norm_emittance_initial / linac.beam.norm_emittance_y
            linac.damping_ring.num_rings = max(0, np.ceil(np.log(linac.damping_ring.damping_ratio)/self.max_damping_times_per_ring))
            if linac.damping_ring.num_rings > 0:
                linac.damping_ring.energy = linac.beam.energy_initial
                linac.damping_ring.bunch_separation = 10e-9 # [s]
                linac.damping_ring.Bfield_avg = 0.2 # [T]
                linac.damping_ring.relative_energy_loss_per_turn = 0.001
                linac.damping_ring.bend_radius_min = linac.damping_ring.energy/(SI.c*linac.damping_ring.Bfield_avg)
                linac.damping_ring.circumference = max(2*np.pi*linac.damping_ring.bend_radius_min, self.bunch_pattern.bunches_per_train*SI.c*linac.damping_ring.bunch_separation)
                linac.damping_ring.bend_radius = linac.damping_ring.circumference / (2*np.pi)
                linac.damping_ring.time_per_turn = linac.damping_ring.circumference / SI.c
                linac.damping_ring.energy_loss_per_turn = linac.damping_ring.relative_energy_loss_per_turn * linac.damping_ring.energy
                linac.damping_ring.characteristic_damping_time = linac.damping_ring.time_per_turn / linac.damping_ring.relative_energy_loss_per_turn
                linac.damping_ring.characteristic_damping_time_vert = 2*linac.damping_ring.characteristic_damping_time
                linac.damping_ring.num_damping_times = np.log(linac.damping_ring.damping_ratio) / linac.damping_ring.num_rings
                linac.damping_ring.total_damping_time = linac.damping_ring.num_damping_times * linac.damping_ring.characteristic_damping_time
                linac.damping_ring.power_emitted = self.bunch_pattern.bunches_per_train * linac.beam.charge * linac.damping_ring.energy_loss_per_turn / linac.damping_ring.time_per_turn
            else:
                linac.damping_ring.circumference = 0
                linac.damping_ring.total_damping_time = 0
                linac.damping_ring.bend_radius = 0

        
        ## BUNCH TRAIN PATTERN
        
        # lower bunch train repetition rate if beyond the emittance damping time
        self.bunch_pattern.reprate_trains = 1/np.max([1/self.bunch_pattern.reprate_trains, self.linac1.damping_ring.total_damping_time, self.linac2.damping_ring.total_damping_time])
        
        # repetition rate
        self.bunch_pattern.reprate_avg = self.bunch_pattern.reprate_trains*self.bunch_pattern.bunches_per_train
        
        # first estimate of max linac field and breakdowns (normal-conducting RF)
        self.bunch_pattern.max_accel_gradient = 1e6*(1e28*self.BDR_max)**(1/30)*(self.bunch_pattern.bunches_per_train*self.bunch_pattern.bunch_spacing)**(-1/6)
        
        
        ## SET UP THE LINACS
        
        # go through linacs separately
        i = 0
        for linac in [self.linac1, self.linac2]:
            i = i+1
            
            ## SET UP AS PWFA ELECTRON LINAC
            if linac.pwfa_linac.enable:

                # no RF linac
                linac.rf_linac.length = 0
                linac.rf_linac.wallplug_power = 0
                
                ## PWFA PARAMETERS
                
                # driver peak field and charge
                linac.pwfa_linac.decel_gradient_peak = self.pwfa.accel_gradient / self.pwfa.transformer_ratio
                linac.pwfa_linac.driver.charge = linac.beam.charge * self.pwfa.transformer_ratio / (self.pwfa.extraction_efficiency * self.pwfa.depletion_efficiency)
                
                # if using combined driver/positron linac: recalculate number of stages required
                if linac == self.linac1 and self.combined_driver_positron_linac:
                    linac.pwfa_linac.driver.energy = self.linac2.beam.energy_final
                    self.pwfa.num_stages = int(np.ceil((self.linac1.beam.energy_final - self.linac1.beam.energy_initial) / (self.pwfa.transformer_ratio * linac.pwfa_linac.driver.energy)))
                
                # plasma wake longitudinal energy densities
                linac.pwfa_linac.energy_density_z_extracted = linac.beam.charge*self.pwfa.accel_gradient
                linac.pwfa_linac.energy_density_z_wake = linac.pwfa_linac.energy_density_z_extracted/self.pwfa.extraction_efficiency
                linac.pwfa_linac.energy_density_z_remaining = linac.pwfa_linac.energy_density_z_wake - linac.pwfa_linac.energy_density_z_extracted
                
                # blowout radius
                linac.pwfa_linac.norm_blowout_radius = ((32*SI.r_e/(SI.m_e*SI.c**2))*linac.pwfa_linac.energy_density_z_wake)**(1/4)
                
                # optimal wakefield loading (finding the plasma density)
                linac.pwfa_linac.norm_accel_gradient = 1/3 * (linac.pwfa_linac.norm_blowout_radius)**1.15
                linac.pwfa_linac.wavebreaking_field = self.pwfa.accel_gradient / linac.pwfa_linac.norm_accel_gradient
                linac.pwfa_linac.plasma_wavenumber = linac.pwfa_linac.wavebreaking_field/(SI.m_e*SI.c**2/SI.e)
                linac.pwfa_linac.plasma_density = linac.pwfa_linac.plasma_wavenumber**2*SI.m_e*SI.c**2*SI.epsilon_0/SI.e**2
                
                # stage length
                linac.pwfa_linac.num_stages = int(self.pwfa.num_stages)
                linac.pwfa_linac.stage.energy_gain = (linac.beam.energy_final-linac.beam.energy_initial) / linac.pwfa_linac.num_stages
                linac.pwfa_linac.stage.length = linac.pwfa_linac.stage.energy_gain / self.pwfa.accel_gradient
                linac.pwfa_linac.length_stages_total = linac.pwfa_linac.stage.length * linac.pwfa_linac.num_stages
                        
                # if not using combined driver/positron linac: calculate driver energy
                if not self.combined_driver_positron_linac:
                    linac.pwfa_linac.driver.energy = linac.pwfa_linac.decel_gradient_peak * linac.pwfa_linac.stage.length
                
                # driver parameters
                linac.pwfa_linac.driver.bunch_length = (2/3)*self.pwfa.transformer_ratio/linac.pwfa_linac.plasma_wavenumber
                linac.pwfa_linac.driver.peak_current = linac.pwfa_linac.driver.charge/(np.sqrt(2*np.pi)*linac.pwfa_linac.driver.bunch_length/SI.c)
        
                # interstage length
                linac.pwfa_linac.interstage.energies = linac.beam.energy_initial + linac.pwfa_linac.stage.energy_gain * np.arange(linac.pwfa_linac.num_stages)
                linac.pwfa_linac.length_interstages_total = np.sum(self.pwfa.interstage_length_scaling * np.sqrt(linac.pwfa_linac.interstage.energies[1:]))
                
                # total plasma linac lench
                linac.pwfa_linac.length = linac.pwfa_linac.length_interstages_total + linac.pwfa_linac.length_stages_total
                
                # cooling requirement
                linac.pwfa_linac.stage.cooling_per_length = linac.pwfa_linac.energy_density_z_remaining * self.bunch_pattern.bunches_per_train * self.bunch_pattern.reprate_trains 
                
                # matching
                linac.beam.beta_matched_initial = np.sqrt(2*linac.beam.gamma_initial)/linac.pwfa_linac.plasma_wavenumber
                linac.beam.beta_matched_final = np.sqrt(2*linac.beam.gamma_final)/linac.pwfa_linac.plasma_wavenumber
                linac.beam.sigx_matched_initial = np.sqrt(linac.beam.beta_matched_initial*linac.beam.norm_emittance_x/linac.beam.gamma_initial)
                linac.beam.sigy_matched_initial = np.sqrt(linac.beam.beta_matched_initial*linac.beam.norm_emittance_y/linac.beam.gamma_initial)
                
                # transverse instability
                linac.pwfa_linac.norm_trans_wakefield = (self.pwfa.extraction_efficiency**2/4)/(1-self.pwfa.extraction_efficiency)
                linac.pwfa_linac.phase_advance = np.sqrt(2)*(np.sqrt(linac.beam.gamma_final)-np.sqrt(linac.beam.gamma_initial))/linac.pwfa_linac.norm_accel_gradient
                linac.pwfa_linac.instability_parameter = linac.pwfa_linac.phase_advance * linac.pwfa_linac.norm_trans_wakefield

                # emittance growth from transverse instability (multiple uncorrelated stages) [from Lebedev et al. PRAB 2017]
                linac.pwfa_linac.emittance_growth_instability = 0
                for energy in linac.pwfa_linac.interstage.energies:
                    gamma_stage_initial = energy/(SI.m_e*SI.c**2/SI.e)
                    gamma_stage_final = (energy+linac.pwfa_linac.stage.energy_gain)/(SI.m_e*SI.c**2/SI.e)
                    phase_advance_stage = np.sqrt(2)*(np.sqrt(gamma_stage_final)-np.sqrt(gamma_stage_initial))/linac.pwfa_linac.norm_accel_gradient
                    instability_parameter_stage = phase_advance_stage * linac.pwfa_linac.norm_trans_wakefield
                    amplitude_growth_stage = np.exp(instability_parameter_stage**2/(60+2.2*instability_parameter_stage**1.57))
                    
                    # add emittance in squares (random walk): to be checked in simulations
                    beta_matched_stage_initial = np.sqrt(2*gamma_stage_initial)/linac.pwfa_linac.plasma_wavenumber
                    emittance_growth_instability_stage = amplitude_growth_stage**2 * self.pwfa.driver_jitter_transverse**2 * gamma_stage_initial / (2*beta_matched_stage_initial)
                    
                    #linac.pwfa_linac.emittance_growth_instability = np.sqrt(linac.pwfa_linac.emittance_growth_instability**2 + emittance_growth_instability_stage**2)
                    linac.pwfa_linac.emittance_growth_instability = linac.pwfa_linac.emittance_growth_instability + emittance_growth_instability_stage
                
                # add emittance to beam
                linac.beam.norm_emittance_x_final = np.sqrt(linac.beam.norm_emittance_x**2 + linac.pwfa_linac.emittance_growth_instability**2)
                linac.beam.norm_emittance_y_final = np.sqrt(linac.beam.norm_emittance_y**2 + linac.pwfa_linac.emittance_growth_instability**2)

                # final beam size
                linac.beam.sigx_matched_final = np.sqrt(linac.beam.beta_matched_final*linac.beam.norm_emittance_x_final/linac.beam.gamma_final)
                linac.beam.sigy_matched_final = np.sqrt(linac.beam.beta_matched_final*linac.beam.norm_emittance_y_final/linac.beam.gamma_final)

                
                ## PWFA DRIVER LINAC
                
                # driver linac beam power
                linac.pwfa_linac.driver_linac.energy_per_driver_train = linac.pwfa_linac.driver.charge * linac.pwfa_linac.driver.energy * linac.pwfa_linac.num_stages * self.bunch_pattern.bunches_per_train
                linac.pwfa_linac.driver_linac.peak_beam_power_per_length = self.driver_linac.peak_power_klystrons/self.driver_linac.cell_length
                linac.pwfa_linac.driver_linac.beam_power_avg = linac.pwfa_linac.driver.charge * linac.pwfa_linac.driver.energy * linac.pwfa_linac.num_stages * self.bunch_pattern.reprate_avg
                
                # set lower driver linac gradient if inconsistent with breakdown rate
                max_field_guess = (1e6*(1e28*self.BDR_max)**(1/30)*((linac.pwfa_linac.driver.charge*(linac.pwfa_linac.num_stages+int(self.combined_driver_positron_linac))*self.bunch_pattern.bunches_per_train)/linac.pwfa_linac.driver_linac.peak_beam_power_per_length)**(-1/6))**(6/7)
                linac.pwfa_linac.driver_linac.accel_gradient = min(self.driver_linac.accel_gradient, max_field_guess)

                # driver linac length
                linac.pwfa_linac.driver_linac.length = linac.pwfa_linac.driver.energy / linac.pwfa_linac.driver_linac.accel_gradient

                # number of klystrons: TODO: cost scaling with number of klystrons
                linac.pwfa_linac.driver_linac.num_klystrons = linac.pwfa_linac.driver_linac.length/self.driver_linac.cell_length
                
                # full train duration
                linac.pwfa_linac.driver_linac.driver_spacing = self.driver_linac.accel_gradient * linac.pwfa_linac.driver.charge / linac.pwfa_linac.driver_linac.peak_beam_power_per_length
                if self.combined_driver_positron_linac: # add one for the positron bunch
                    num_bunches = linac.pwfa_linac.num_stages + 1
                else:
                    num_bunches = linac.pwfa_linac.num_stages
                linac.pwfa_linac.driver_linac.singletrain_duration = num_bunches * linac.pwfa_linac.driver_linac.driver_spacing
                linac.pwfa_linac.driver_linac.fulltrain_duration = linac.pwfa_linac.driver_linac.singletrain_duration * self.bunch_pattern.bunches_per_train

                # increase the bunch spacing (colling beams) if driver train too long
                if self.bunch_pattern.bunch_spacing < linac.pwfa_linac.driver_linac.singletrain_duration:
                    self.bunch_pattern.bunch_spacing = linac.pwfa_linac.driver_linac.singletrain_duration
                
                # driver linac field and breakdowns
                self.bunch_pattern.max_accel_gradient = 1e6*(1e28*self.BDR_max)**(1/30)*linac.pwfa_linac.driver_linac.fulltrain_duration**(-1/6)
                
                # driver linac efficiency
                linac.pwfa_linac.driver_linac.norm_R_upon_Q = 1/(SI.c*SI.epsilon_0)*(self.driver_linac.cavity_frequency/SI.c)
                linac.pwfa_linac.driver_linac.cavity_energy_per_length = linac.pwfa_linac.driver_linac.accel_gradient**2/(2*np.pi*self.driver_linac.cavity_frequency*linac.pwfa_linac.driver_linac.norm_R_upon_Q)
                linac.pwfa_linac.driver_linac.filling_time = 2 * linac.pwfa_linac.driver_linac.cavity_energy_per_length * self.driver_linac.cell_length / self.driver_linac.peak_power_klystrons
                linac.pwfa_linac.driver_linac.beamloading_efficiency = 1/(1 + linac.pwfa_linac.driver_linac.filling_time/linac.pwfa_linac.driver_linac.fulltrain_duration)
                linac.pwfa_linac.driver_linac.efficiency_total = self.driver_linac.efficiency_wallplug_to_rf * linac.pwfa_linac.driver_linac.beamloading_efficiency
                
                # power per klystron
                linac.pwfa_linac.driver_linac.wallplug_power = linac.pwfa_linac.driver_linac.beam_power_avg/linac.pwfa_linac.driver_linac.efficiency_total
                linac.pwfa_linac.driver_linac.power_per_klystron_avg = linac.pwfa_linac.driver_linac.wallplug_power / linac.pwfa_linac.driver_linac.num_klystrons
                
                
                ## PWFA KICKER TREE
                
                # add PWFA kicker tree
                ds_per_stage_average = (linac.pwfa_linac.length-linac.pwfa_linac.stage.length)/linac.pwfa_linac.num_stages
                pathlength_difference = linac.pwfa_linac.driver_linac.driver_spacing * SI.c
                linac.pwfa_linac.kicker_tree.angle = np.arccos(1/(1+pathlength_difference/ds_per_stage_average))

                # calculate the total length of kicker tree beamline
                linac.pwfa_linac.kicker_tree.length_total = 0
                linac.pwfa_linac.kicker_tree.ss_start_stage = np.empty_like(range(linac.pwfa_linac.num_stages))
                linac.pwfa_linac.kicker_tree.ss_end_stage = np.empty_like(range(linac.pwfa_linac.num_stages))
                linac.pwfa_linac.kicker_tree.dxs_stage = np.empty_like(range(linac.pwfa_linac.num_stages))
                for i in range(linac.pwfa_linac.num_stages):

                    # find which group the stage and kicker belongs to
                    stage_group = np.floor(i/self.pwfa.kickers_per_side)
                    stage_in_group = np.mod(i, self.pwfa.kickers_per_side)
                    kicker_side = 2*np.mod(stage_group, 2)-1
                    first_stage_in_group = i-stage_in_group

                    # calculate start and end of the kicker beamline for that stage
                    linac.pwfa_linac.kicker_tree.ss_start_stage[i] = first_stage_in_group*linac.pwfa_linac.stage.length + np.sum(self.pwfa.interstage_length_scaling*np.sqrt(linac.pwfa_linac.interstage.energies[0:int(first_stage_in_group)]))
                    linac.pwfa_linac.kicker_tree.ss_end_stage[i] = (i+1)*linac.pwfa_linac.stage.length + np.sum(self.pwfa.interstage_length_scaling*np.sqrt(linac.pwfa_linac.interstage.energies[0:(i+1)]))

                    # transverse and longitudinal offsets
                    ds_stage = linac.pwfa_linac.kicker_tree.ss_end_stage[i]-linac.pwfa_linac.kicker_tree.ss_start_stage[i]
                    dx_stage = kicker_side*(ds_stage/2)*np.tan(linac.pwfa_linac.kicker_tree.angle)
                    linac.pwfa_linac.kicker_tree.dxs_stage[i] = dx_stage

                    # add beamline length
                    length_beamline_straights = np.sqrt(ds_stage**2 + dx_stage**2)/2
                    linac.pwfa_linac.kicker_tree.length_total += length_beamline_straights                    
                    if not np.mod(i+1, self.pwfa.kickers_per_side) or i+1 == linac.pwfa_linac.num_stages:
                        linac.pwfa_linac.kicker_tree.length_total += length_beamline_straights # in the last stage per group, add the common length

            
            ## SET UP AS RF LINAC
            else: 

                # no plasma driver linac
                linac.pwfa_linac.length_stages_total = 0
                linac.pwfa_linac.length_interstages_total = 0
                linac.pwfa_linac.length = 0
                linac.pwfa_linac.driver_linac.length = 0
                linac.pwfa_linac.driver_linac.wallplug_power = 0
                linac.pwfa_linac.driver_linac.beam_power_avg = 0
                linac.pwfa_linac.kicker_tree.length_total = 0

                # assume no emittance growth
                linac.beam.norm_emittance_x_final = linac.beam.norm_emittance_x
                linac.beam.norm_emittance_y_final = linac.beam.norm_emittance_y

               
                # account for cool copper (C^3 cavities) or superconducting RF
                if self.use_coolcopper_rf: 
                    field_multiplier = np.sqrt(2.5)
                    linac.rf_linac.cavity_frequency = 5.7e9 # [Hz]
                    linac.rf_linac.operating_temperature = 77 # [K]
                    if self.energy_centerofmass >= 500e9:
                        linac.rf_linac.cell_length = 1.67 # [m]
                    else:
                        linac.rf_linac.cell_length = 0.625 # [m]
                    linac.rf_linac.peak_power_klystrons = 50e6
                    linac.rf_linac.efficiency_wallplug_to_rf = self.ccrf_efficiency_wallplug_to_rf
                elif self.use_superconducting_rf:
                    field_multiplier = np.sqrt(4)
                    linac.rf_linac.cavity_frequency = 1.3e9 # [Hz]
                    linac.rf_linac.operating_temperature = 4 # [K]
                    linac.rf_linac.cell_length = 9 # [m]
                    linac.rf_linac.peak_power_klystrons = 10e6
                    linac.rf_linac.efficiency_wallplug_to_rf = self.scrf_efficiency_wallplug_to_rf
                else:
                    field_multiplier = 1
                    linac.rf_linac.cavity_frequency = 2e9 # [Hz]
                    linac.rf_linac.operating_temperature = 300 # [K]
                    linac.rf_linac.cell_length = 2 # [m]
                    linac.rf_linac.peak_power_klystrons = 50e6
                    linac.rf_linac.efficiency_wallplug_to_rf = self.ncrf_efficiency_wallplug_to_rf
                  
                # rf linac
                if self.combined_driver_positron_linac:
                    linac.rf_linac.length = 0
                    linac.rf_linac.wallplug_power = 0
                else:
                    
                    linac.rf_linac.accel_gradient = 0.9 * self.bunch_pattern.max_accel_gradient * field_multiplier
                    
                    # RF linac efficiency
                    linac.rf_linac.norm_R_upon_Q = 1/(SI.c*SI.epsilon_0)*(linac.rf_linac.cavity_frequency/SI.c)
                    linac.rf_linac.cavity_energy_per_length = linac.rf_linac.accel_gradient**2/(2*np.pi*linac.rf_linac.cavity_frequency*linac.rf_linac.norm_R_upon_Q)
                    linac.rf_linac.filling_time = 2 * linac.rf_linac.cavity_energy_per_length * linac.rf_linac.cell_length / linac.rf_linac.peak_power_klystrons
                    linac.rf_linac.beamloading_efficiency = 1/(1 + linac.rf_linac.filling_time/(self.bunch_pattern.bunch_spacing * self.bunch_pattern.bunches_per_train))
                    room_temperature = 300 # [K] for Carnot engine efficiency
                    linac.rf_linac.cooling_efficiency = linac.rf_linac.operating_temperature/room_temperature
                    linac.rf_linac.efficiency_total = linac.rf_linac.efficiency_wallplug_to_rf / (1/linac.rf_linac.beamloading_efficiency +  (1-linac.rf_linac.beamloading_efficiency)/linac.rf_linac.cooling_efficiency)
                    
                    linac.rf_linac.length = linac.beam.energy_final / linac.rf_linac.accel_gradient
                    linac.rf_linac.power_per_length = linac.rf_linac.accel_gradient * linac.beam.charge / self.bunch_pattern.bunch_spacing
                    #linac.rf_linac.wallplug_power = linac.beam.charge * linac.beam.energy_final * self.bunch_pattern.reprate_avg / (self.eta_prod_positrons * efficiency_multiplier)
                    linac.rf_linac.wallplug_power = linac.beam.charge * linac.beam.energy_final * self.bunch_pattern.reprate_avg / linac.rf_linac.efficiency_total


            ## INJECTOR LINACS
            linac.injector.length = linac.beam.energy_initial / self.bunch_pattern.max_accel_gradient
            linac.injector.wallplug_power = linac.beam.charge * linac.beam.energy_initial * self.bunch_pattern.reprate_avg / self.eta_prod_positrons
            
            ## IP BEAMS
            linac.sigx_IP = np.sqrt(self.betaxIP_min*linac.beam.norm_emittance_x_final/linac.beam.gamma_final)
            linac.sigy_IP = np.sqrt(self.betayIP_min*linac.beam.norm_emittance_y_final/linac.beam.gamma_final)
            
            ## POWER
            linac.wallplug_power = linac.pwfa_linac.driver_linac.wallplug_power + linac.rf_linac.wallplug_power + linac.injector.wallplug_power

            ## LENGTHS
            linac.length = linac.pwfa_linac.driver_linac.length + linac.pwfa_linac.length + linac.rf_linac.length

            ## COSTS
            linac.pwfa_linac.cost = linac.pwfa_linac.length_stages_total * self.cost.per_length.plasma_stage + linac.pwfa_linac.length_interstages_total * self.cost.per_length.plasma_interstage + linac.pwfa_linac.kicker_tree.length_total * self.cost.per_length.plasma_kicker_beamline
            linac.pwfa_linac.driver_linac.cost = linac.pwfa_linac.driver_linac.length * self.cost.per_length.rf_linac
            linac.rf_linac.cost = linac.rf_linac.length * self.cost.per_length.rf_linac
            linac.injector.cost = linac.injector.length * self.cost.per_length.rf_linac
            linac.cost = linac.pwfa_linac.cost + linac.pwfa_linac.driver_linac.cost + linac.rf_linac.cost + linac.injector.cost

            ## DUMPED POWER
            linac.beam.average_power = linac.beam.energy_final * linac.beam.charge * self.bunch_pattern.reprate_avg
            linac.dumped_beam_power = linac.beam.average_power + linac.pwfa_linac.driver_linac.beam_power_avg*(1-self.pwfa.depletion_efficiency)

        
        
        ## BEAM DELIVERY SYSTEM
        self.linac1.bds.length = np.sqrt(self.linac1.beam.energy_final/500e9)*2.25e3
        self.linac2.bds.length = np.sqrt(self.linac2.beam.energy_final/500e9)*2.25e3
        if self.combined_driver_positron_linac:
            self.linac2.bds.length = self.linac2.bds.length*0.25

        
        ## IP and LUMINOSITY
        self.sigx_IP_max = max(self.linac1.sigx_IP, self.linac2.sigx_IP)
        self.sigy_IP_max = max(self.linac1.sigy_IP, self.linac2.sigy_IP)
        self.lumi_geo = self.bunch_pattern.reprate_avg * self.linac1.beam.num_particles * self.linac2.beam.num_particles / (4*np.pi*self.sigx_IP_max*self.sigy_IP_max)
        # TODO: account for beam-beam effects
        # TODO: account for gamma–gamma conversion and physics multiplier
        
        # turn-arounds (if combined-function linac)
        self.turn_around = SimpleNamespace()
        self.turn_around.num_turn_arounds = int(self.use_driver_turnaround)+int(self.combined_driver_positron_linac)
        self.turn_around.Bfield_avg = 1 # [T]
        self.turn_around.bend_radius = self.linac2.beam.energy_final/(SI.c*self.turn_around.Bfield_avg)
        self.turn_around.circumference = 2*np.pi*self.turn_around.bend_radius
        self.turn_around.total_length = self.turn_around.num_turn_arounds*self.turn_around.circumference
        # TODO: account for power loss for high energies in strong bends

        
        ## TOTAL POWERS
        
        # total power
        self.P_per_damping_ring = 10e6 # [W] TODO: scale with the emitted power
        self.P_damping_rings = (self.linac1.damping_ring.num_rings + self.linac2.damping_ring.num_rings) * self.P_per_damping_ring
        self.P_driver_linacs = self.linac1.pwfa_linac.driver_linac.wallplug_power + self.linac2.pwfa_linac.driver_linac.wallplug_power
        self.P_rf_linacs = self.linac1.rf_linac.wallplug_power + self.linac2.rf_linac.wallplug_power
        self.P_injectors = self.linac1.injector.wallplug_power + self.linac2.injector.wallplug_power
        self.P_linacs = self.P_driver_linacs + self.P_injectors + self.P_rf_linacs
        self.P_overheads = self.P_linacs * self.overheads.power.total
        self.P_collider = self.P_linacs + self.P_damping_rings + self.P_overheads


        ## LUMINOSITY AND PHYSICS COST
        
        # luminosity per power
        self.lumi_per_power = self.lumi_geo/self.P_collider

        # required luminosity at this energy
        if self.linac1.beam.charge_sign*self.linac2.beam.charge_sign < 0: # electron-positron collider
            self.integrated_lumi_required = np.sqrt(self.integrated_lumi_required_minimum**2 + (self.integrated_lumi_required_per_com_energy*(self.energy_centerofmass))**2)
        else: # gamma-gamma collider
            self.integrated_lumi_required = self.integrated_lumi_required_gamma_gamma_model(self.energy_centerofmass)
        
        # power cost
        self.time_collisions = self.integrated_lumi_required/self.lumi_geo
        self.total_energy_collisions = self.P_collider*self.time_collisions
        self.cost_power = self.total_energy_collisions*self.cost_per_power


        ## TOTAL LENGTHS

        # linac lengths
        self.length_plasma_linacs = self.linac1.pwfa_linac.length + self.linac2.pwfa_linac.length
        self.length_driver_linacs = self.linac1.pwfa_linac.driver_linac.length + self.linac2.pwfa_linac.driver_linac.length
        self.length_main_rf_linacs = self.linac1.rf_linac.length + self.linac2.rf_linac.length
        self.length_injectors = self.linac1.injector.length + self.linac2.injector.length
        
        # end-to-end collider length
        if self.use_driver_turnaround and self.combined_driver_positron_linac:
            self.length_end_to_end = max(self.linac1.pwfa_linac.length + self.linac1.bds.length + self.linac2.bds.length + 2*self.turn_around.bend_radius, 
                                         self.linac1.pwfa_linac.driver_linac.length + 3*self.turn_around.bend_radius + self.linac1.damping_ring.bend_radius)
        elif self.use_driver_turnaround and not self.combined_driver_positron_linac:
            self.length_end_to_end = max(self.linac1.pwfa_linac.length + self.linac2.length + self.linac1.bds.length + self.linac2.bds.length + self.turn_around.bend_radius + self.linac1.damping_ring.bend_radius, 
                                         self.linac1.pwfa_linac.driver_linac.length + 3*self.turn_around.bend_radius + self.linac1.damping_ring.bend_radius)
        elif not self.use_driver_turnaround and self.combined_driver_positron_linac:
            self.length_end_to_end = self.linac1.length + self.linac1.bds.length + self.linac2.bds.length + self.linac1.damping_ring.bend_radius + self.turn_around.bend_radius
        elif not self.use_driver_turnaround and not self.combined_driver_positron_linac:
            self.length_end_to_end = self.linac1.length + self.linac2.length + self.linac1.bds.length + self.linac2.bds.length + self.linac1.damping_ring.bend_radius
        
        # tunnel cost
        if self.combined_damping_rings:
            self.length_tunnel_damping_rings = self.linac1.damping_ring.circumference
        else:
            self.length_tunnel_damping_rings = self.linac1.damping_ring.circumference + self.linac2.damping_ring.circumference
        self.length_tunnel = self.linac1.length + self.linac1.bds.length + self.linac2.length + self.linac2.bds.length + self.length_tunnel_damping_rings + self.turn_around.total_length 
        self.cost_tunnel = self.length_tunnel * self.cost.per_length.tunnel
        
        # power infrastructure cost
        self.cost_power_infrastructure = self.cost_per_power_infrastructure_linacs * self.P_rf_linacs + self.cost_per_power_infrastructure_other * (self.P_collider-self.P_rf_linacs)
        
        # particle sources
        self.cost_particle_sources = 3 * self.cost_per_source # TODO: scale with number of sources

        # damping ring costs
        self.length_damping_rings = self.linac1.damping_ring.num_rings * self.linac1.damping_ring.circumference + self.linac2.damping_ring.num_rings * self.linac2.damping_ring.circumference
        self.cost_damping_rings = self.length_damping_rings*self.cost.per_length.damping_ring
        
        # rf linac costs
        self.cost_driver_linacs = self.linac1.pwfa_linac.driver_linac.cost + self.linac2.pwfa_linac.driver_linac.cost
        self.cost_main_rf_linacs = self.linac1.rf_linac.cost + self.linac2.rf_linac.cost
        self.cost_injector_linacs = self.linac1.injector.cost + self.linac2.injector.cost
        self.cost_all_rf_linacs = self.cost_driver_linacs + self.cost_main_rf_linacs + self.cost_injector_linacs

        # plasma linac costs
        self.cost_pwfa_linacs = self.linac1.pwfa_linac.cost + self.linac2.pwfa_linac.cost
        
        # turn-around cost
        self.cost_turn_arounds = self.cost.per_length.turn_arounds * self.turn_around.total_length
        
        # BDS cost
        self.cost_BDS = self.cost.per_length.BDS * (self.linac1.bds.length + self.linac2.bds.length)

        # IP cost
        self.num_IPs = 1
        self.cost_IP = self.cost_per_IP*self.num_IPs
        
        # dumps
        self.beam_power_dumped_total = self.linac1.dumped_beam_power + self.linac2.dumped_beam_power
        self.cost_dumps = self.cost_per_beam_power_dumps * self.beam_power_dumped_total
        
        # construction cost
        self.cost_construction = self.cost_all_rf_linacs + self.cost_tunnel + self.cost_damping_rings + self.cost_pwfa_linacs + self.cost_BDS + self.cost_IP + self.cost_power_infrastructure + self.cost_particle_sources + self.cost_dumps + self.cost_turn_arounds
        self.overheads.total = self.overheads.construction.design_and_development + self.overheads.construction.controls_and_cabling + self.overheads.construction.installation_and_commissioning + self.overheads.construction.management_inspection
        self.cost_overheads = self.cost_construction*self.overheads.total

        # uptime costs
        self.total_runtime = self.time_collisions / self.uptime_percentage
        self.labor_maintenance_per_year = self.maintenance_labor_per_construction_cost * self.cost_construction
        self.cost_maintenance = self.labor_maintenance_per_year*self.cost_per_labor*(self.total_runtime/(365*24*3600))
        
        # combined programme cost
        self.cost_full_programme = self.cost_construction + self.cost_overheads + self.cost_power + self.cost_maintenance

    
        ## EMISSIONS (GLOBAL WARMING POTENTIAL)
        self.GWP_tunnels = self.length_tunnel * self.emissions_per_tunnel_length
        self.GWP_power = self.total_energy_collisions * self.emissions_per_energy_usage
        self.GWP_total = self.GWP_tunnels + self.GWP_power
        
        # full program cost with carbon tax
        self.cost_carbon_tax = self.carbon_tax_per_emissions * self.GWP_total
        self.cost_full_programme_with_carbon_tax = self.cost_full_programme + self.cost_carbon_tax



    ## PRINTING FUNCTIONS
    
    def print_all(self):
        self.print_bunches()
        self.print_plasma_wake()
        self.print_drive_bunches()
        self.print_plasma_linac()
        self.print_driver_linac()
        self.print_main_rf_linac()
        self.print_damping_rings()
        self.print_luminosity()
        self.print_power()
        self.print_lengths()
        self.print_costs()

    def print_bunches(self):
        print('\nELECTRON/POSITRON BUNCH PARAMETERS:')
        print('>> Electron final energy = {:.1f} GeV'.format(self.linac1.beam.energy_final/1e9))
        print('>> Positron final energy = {:.1f} GeV'.format(self.linac2.beam.energy_final/1e9))
        print('>> Electron charge = {:.2f} nC'.format(self.linac1.beam.charge/1e-9))
        print('>> Positron charge = {:.2f} nC'.format(self.linac2.beam.charge/1e-9))
        print('>> Electron norm. emittance = {:.1f} x {:.2f} mm mrad'.format(self.linac1.beam.norm_emittance_x/1e-6, self.linac1.beam.norm_emittance_y/1e-6))
        print('>> Positron norm. emittance = {:.1f} x {:.2f} mm mrad'.format(self.linac2.beam.norm_emittance_x/1e-6, self.linac2.beam.norm_emittance_y/1e-6))
        print('>> Number of bunches in train = {:.0f}'.format(self.bunch_pattern.bunches_per_train))
        print('>> Bunch separation = {:.1f} ns'.format(self.bunch_pattern.bunch_spacing/1e-9))
        print('>> Bunch train rep. rate = {:.1f} Hz'.format(self.bunch_pattern.reprate_trains))
        print('>> Average collision rate = {:.1f} kHz'.format(self.bunch_pattern.reprate_avg/1e3))
    
    def print_plasma_wake(self):
        for i, linac in enumerate([self.linac1, self.linac2]):
            if linac.pwfa_linac.enable:
                print('\nPLASMA WAKE PARAMETERS:')
                print('>> Accelerating field          = {:.2f} GV/m'.format(self.pwfa.accel_gradient/1e9))
                print('>> Wake-to-beam efficiency     = {:.0f}%'.format(self.pwfa.extraction_efficiency/1e-2))
                print('>> Driver depletion efficiency = {:.0f}%'.format(self.pwfa.depletion_efficiency/1e-2))
                print('>> Longitudinal energy density = {:.1f} J/m'.format(linac.pwfa_linac.energy_density_z_wake))
                print('>> Energy absorbed per length  = {:.1f} J/m'.format(linac.pwfa_linac.energy_density_z_extracted))
                print('>> Energy remaining per length = {:.1f} J/m'.format(linac.pwfa_linac.energy_density_z_remaining))
                print('>> Peak decelerating field     = {:.2f} GV/m'.format(linac.pwfa_linac.decel_gradient_peak/1e9))
                print('>> Blowout radius              = {:.2f} kp^-1'.format(linac.pwfa_linac.norm_blowout_radius))
                print('>> Plasma density              = {:.3g} cm^-3'.format(linac.pwfa_linac.plasma_density/1e6))
                print('>> Plasma skin depth           = {:.0f} µm'.format((1/linac.pwfa_linac.plasma_wavenumber)/1e-6))
                print('>> Normalized field strength   = {:.2f}'.format(linac.pwfa_linac.norm_accel_gradient))
                print('>> Wavebreaking field          = {:.2f} GV/m'.format(linac.pwfa_linac.wavebreaking_field/1e9))
                print('>> Norm. transverse wakefield  = {:.3f}'.format(linac.pwfa_linac.norm_trans_wakefield))

    def print_drive_bunches(self):
        for i, linac in enumerate([self.linac1, self.linac2]):
            if linac.pwfa_linac.enable:
                print('\nDRIVE BUNCH PARAMETERS:')
                print('>> Driver charge       = {:.2f} nC'.format(linac.pwfa_linac.driver.charge/1e-9))
                print('>> Driver energy       = {:.1f} GeV'.format(linac.pwfa_linac.driver.energy/1e9))
                print('>> Driver bunch length = {:.1f} µm rms'.format(linac.pwfa_linac.driver.bunch_length/1e-6))
                print('>> Driver peak current = {:.1f} kA'.format(linac.pwfa_linac.driver.peak_current/1e3))
        
    def print_plasma_linac(self):
        for i, linac in enumerate([self.linac1, self.linac2]):
            if linac.pwfa_linac.enable:
                print('\nPLASMA LINAC:')
                print('>> Number of stages = {:.0f}'.format(linac.pwfa_linac.num_stages))
                print('>> Stage length = {:.1f} m'.format(linac.pwfa_linac.stage.length))
                print('>> Energy increase per stage = {:.2f} GeV'.format(linac.pwfa_linac.stage.energy_gain/1e9))
                print('>> Average cooling rate per length in stages = {:.0f} kW/m'.format(linac.pwfa_linac.stage.cooling_per_length/1e3))
                print('>> Matched beta function (initial -> final) = {:.1f} -> {:.1f} cm'.format(linac.beam.beta_matched_initial/1e-2, linac.beam.beta_matched_final/1e-2))
                print('>> Matched beam size (initial -> final) = {:.1f}x{:.1f} µm rms -> {:.1f}x{:.1f} µm rms'.format(linac.beam.sigx_matched_initial/1e-6, linac.beam.sigy_matched_initial/1e-6, linac.beam.sigx_matched_final/1e-6, linac.beam.sigy_matched_final/1e-6))
                print('>> Total phase advance, µ = {:.0f} rad'.format(linac.pwfa_linac.phase_advance))
                print('>> Instability parameter, µ*eta_t = {:.1f}'.format(linac.pwfa_linac.instability_parameter))

    def print_driver_linac(self):
        for i, linac in enumerate([self.linac1, self.linac2]):
            if linac.pwfa_linac.enable:
                print('\nDRIVER RF LINAC:')
                print('>> Energy per driver train = {:.1f} kJ'.format(linac.pwfa_linac.driver_linac.energy_per_driver_train/1e3))
                print('>> Driver beam power (average) = {:.1f} MW'.format(linac.pwfa_linac.driver_linac.beam_power_avg/1e6))
                print('>> Driver linac accelerating gradient = {:.1f} MV/m'.format(linac.pwfa_linac.driver_linac.accel_gradient/1e6))
                print('>> Driver linac peak power per length = {:.1f} MW/m'.format(linac.pwfa_linac.driver_linac.peak_beam_power_per_length/1e6))
                print('>> Driver bunch separation = {:.1f} ns'.format(linac.pwfa_linac.driver_linac.driver_spacing/1e-9))
                print('>> Driver train duration = {:.0f} µs'.format(linac.pwfa_linac.driver_linac.fulltrain_duration/1e-6))
                print('>> Driver linac length = {:.2f} km'.format(linac.pwfa_linac.driver_linac.length/1e3))
                print('>> Number of driver klystrons = {:.0f}'.format(linac.pwfa_linac.driver_linac.num_klystrons))
                print('>> Driver klystron average power = {:.1f} kW'.format(linac.pwfa_linac.driver_linac.power_per_klystron_avg/1e3))
                print('>> Driver linac average power = {:.1f} MW'.format(linac.pwfa_linac.driver_linac.wallplug_power/1e6))
                print('>> Driver linac efficiency = {:.1f}%'.format(linac.pwfa_linac.driver_linac.efficiency_total*1e2))
                print('>> Driver linac max field (breakdown limit) = {:.0f} MV/m (should be larger than {:.0f} MV/m)'.format(self.bunch_pattern.max_accel_gradient/1e6, linac.pwfa_linac.driver_linac.accel_gradient/1e6))

    def print_main_rf_linac(self):
        for i, linac in enumerate([self.linac1, self.linac2]):
            if not linac.pwfa_linac.enable and not self.combined_driver_positron_linac:
                print('\nMAIN RF LINAC:')
                print('>> RF linac accel. gradient = {:.1f} MV/m'.format(linac.rf_linac.accel_gradient/1e6))
                print('>> RF linac length = {:.1f} km'.format(linac.rf_linac.length/1e3))
    
    def print_damping_rings(self):
        print('\nDAMPING RINGS:')
        if self.linac1.damping_ring.num_rings + self.linac2.damping_ring.num_rings == 0:
             print('>> No damping rings required.')
        else:
            for i, linac in enumerate([self.linac1, self.linac2]):
                if linac.damping_ring.num_rings > 0:
                    print('-- ' + linac.description + ':')
                    print('>> Number of rings = {:.0f}'.format(linac.damping_ring.num_rings))
                    print('>> Damping ring energy = {:.2f} GeV'.format(linac.damping_ring.energy/1e9))
                    print('>> Damping ring circumference = {:.0f} m'.format(linac.damping_ring.circumference))
                    print('>> Damping ring char. damping time = {:.1f} ms'.format(linac.damping_ring.characteristic_damping_time/1e-3))
                    print('>> Damping ring total damping time = {:.1f} ms'.format(linac.damping_ring.total_damping_time/1e-3))
                    print('>> Damping ring emitted power = {:.1f} MW'.format(linac.damping_ring.power_emitted/1e6))
                    
    def print_luminosity(self):
        print('\nLUMINOSITY:')
        print('>> Beam size at IP = {:.1f} x {:.1f} nm rms'.format(self.sigx_IP_max/1e-9, self.sigy_IP_max/1e-9))
        print('>> Geometric luminosity = {:.2g} cm^-2 s^-1'.format(self.lumi_geo/1e4))
        print('>> Luminosity per power = {:.2g} cm^-2 s^-1 MW^-1'.format(self.lumi_per_power/1e-2))
        print('>> Time required for collisions (100% uptime) = {:.1f} years'.format(self.time_collisions/(365*24*3600)))
        print('>> Integrated energy required = {:.1f} TWh'.format(self.total_energy_collisions/(3600*1e12)))

    def print_power(self):
        print('\nPOWER:')
        if not self.combined_driver_positron_linac:
            print('>> Driver RF linac power    = {:.1f} MW'.format(self.P_driver_linacs/1e6))
            print('>> Main beam RF linac power  = {:.1f} MW'.format(self.P_rf_linacs/1e6))
        else:
            print('>> Driver/e+ RF linac power = {:.1f} MW'.format(self.P_driver_linacs/1e6))
        print('>> Injector RF linac power  = {:.1f} MW'.format(self.P_injectors/1e6))
        print('>> Damping ring power       = {:.1f} MW'.format(self.P_damping_rings/1e6))
        print('>> Overhead power ({:.0f}%)     = {:.1f} MW'.format(self.overheads.power.total*1e2, self.P_overheads/1e6))
        print('------------------------------------')
        print('>> Collider wallplug power  = {:.1f} MW'.format(self.P_collider/1e6))
        print('------------------------------------')

    def print_lengths(self):
        print('\nLENGTHS:')
        if self.length_driver_linacs > 0:
            print('>> Driver linac length       = {:.1f} km'.format(self.length_driver_linacs/1e3))
        if self.length_plasma_linacs > 0:
            print('>> Plasma linac length       = {:.1f} km'.format(self.length_plasma_linacs/1e3))
        if self.length_main_rf_linacs > 0:
            print('>> Main RF linac length      = {:.1f} km'.format(self.length_main_rf_linacs/1e3))
        if self.turn_around.num_turn_arounds > 0:
            print('>> Total turnaround length   = {:.1f} km'.format(self.turn_around.total_length/1e3))
        print('>> Electron BDS length       = {:.1f} km'.format(self.linac1.bds.length/1e3))
        print('>> Positron BDS length       = {:.1f} km'.format(self.linac2.bds.length/1e3))
        if self.length_damping_rings > 0:
            print('>> Damping-ring length       = {:.1f} km'.format(self.length_damping_rings/1e3))
        print('>> Total tunnel length       = {:.1f} km'.format(self.length_tunnel/1e3))
        print('----------------------------------------')
        print('>> Collider length (end-to-end) = {:.1f} km'.format(self.length_end_to_end/1e3))
        print('----------------------------------------')
    
    def print_costs(self):
        print('\nCOSTS:')
        print('>> Particle source cost      = {:.0f} MILCU'.format(self.cost_particle_sources/1e6))
        print('>> Damping ring cost         = {:.0f} MILCU'.format(self.cost_damping_rings/1e6))
        print('>> Linac cost                = {:.0f} MILCU'.format((self.cost_all_rf_linacs+self.cost_pwfa_linacs)/1e6))
        print('>> - Driver linacs           = {:.0f} MILCU'.format(self.cost_driver_linacs/1e6))
        print('>> - Plasma linacs           = {:.0f} MILCU'.format(self.cost_pwfa_linacs/1e6))
        if not self.combined_driver_positron_linac:
            print('>> - Main beam linacs        = {:.0f} MILCU'.format(self.cost_main_rf_linacs/1e6))
        print('>> - Injector linacs         = {:.0f} MILCU'.format(self.cost_injector_linacs/1e6))
        print('>> Power infrastructure cost = {:.0f} MILCU'.format(self.cost_power_infrastructure/1e6))
        print('>> Tunnel cost               = {:.0f} MILCU'.format(self.cost_tunnel/1e6))
        print('>> Beam dump cost            = {:.0f} MILCU'.format(self.cost_dumps/1e6))
        if self.combined_driver_positron_linac:
            print('>> Turn-around costs         = {:.0f} MILCU'.format(self.cost_turn_arounds/1e6))
        print('-----------------------------------')
        print('>> Total construction cost       = {:.2f} BILCU'.format(self.cost_construction/1e9))
        print('>> Overhead cost                 = {:.2f} BILCU'.format(self.cost_overheads/1e9))
        print('>> Maintenance cost (for {:.0f} yrs)  = {:.2f} BILCU'.format(self.total_runtime/(365*24*3600), self.cost_maintenance/1e9))
        print('>> Energy cost (for {:.1f}/ab)      = {:.2f} BILCU'.format(self.integrated_lumi_required/1e46, self.cost_power/1e9))
        print('>> Carbon tax (for {:.0f} kton CO2) = {:.0f} MILCU'.format(self.GWP_total/1e3, self.cost_carbon_tax/1e6))
        print('-----------------------------------')
        print('>> ITF cost (excl. run costs)    = {:.2f} BILCU'.format((self.cost_construction+self.cost_overheads)/1e9))
        print('>> Full programme cost           = {:.2f} BILCU'.format(self.cost_full_programme/1e9))
        print('>> Full programme cost + CO2 tax = {:.2f} BILCU'.format(self.cost_full_programme_with_carbon_tax/1e9))
        print('-----------------------------------')


    
    ## VARIATION PLOTS

    # cost variation plots
    def plot_cost_variation(self0, param_name, min=None, max=None, unit='a.u.', scale=1, label=None):

        values = np.logspace(np.log10(min), np.log10(max), 100)
        
        cost_construction = np.empty_like(values)
        cost_overheads = np.empty_like(values)
        cost_power = np.empty_like(values)
        cost_maintenance = np.empty_like(values)
        
        # get original value
        orig_value = self0.get(param_name)
        self = copy.deepcopy(self0)
        
        for i, value in enumerate(values):
            self.set(param_name, value)
            self.calculate_all()
            cost_construction[i] = self.cost_construction
            cost_overheads[i] = self.cost_overheads
            cost_power[i] = self.cost_power
            cost_maintenance[i] = self.cost_maintenance

        # reset values
        self.set(param_name, orig_value)
        self.calculate_all()

        # total cost
        cost_total = cost_construction+cost_overheads+cost_power+cost_maintenance

        # plot everything
        fig, ax = plt.subplots(1,1)
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate((cost_construction/1e9, np.zeros_like(values))), 
                 label='Construction cost')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate((cost_construction/1e9, np.flip(cost_construction+cost_overheads)/1e9)), 
                 label='Overheads')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate(((cost_construction+cost_overheads)/1e9, np.flip(cost_construction+cost_overheads+cost_power)/1e9)), 
                 label='Power cost ({:.1f} inv. ab)'.format(self.integrated_lumi_required/1e46))
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate(((cost_construction+cost_overheads+cost_power)/1e9, np.flip(cost_total)/1e9)), 
                 label='Maintenance cost ({:.0f} FTEs/year)'.format(self0.labor_maintenance_per_year))
        ax.plot(values/scale, cost_total/1e9, 'k-', label='Full programme cost')
        ax.plot([orig_value/scale, orig_value/scale], [0.0, np.max(cost_total/1e9)*1.1], 'k:')
        plt.xscale('log')
        if label is None:
            label = param_name
        plt.xlabel(label + ' (' + unit + ')')
        plt.ylabel('Cost (BILCU)')
        plt.ylim(0, np.max(cost_total/1e9)*1.1)
        plt.xlim(np.min(values)/scale, np.max(values)/scale)
        plt.legend(reverse=True)

    # length variation plots
    def plot_length_variation(self0, param_name, min=None, max=None, unit=None, scale=1, label=None):

        values = np.logspace(np.log10(min), np.log10(max), 100)
        
        length_driver_linac = np.empty_like(values)
        length_plasma_linac = np.empty_like(values)
        length_electron_bds = np.empty_like(values)
        length_positron_bds = np.empty_like(values)
        length_positron_linac = np.empty_like(values)

        orig_value = self0.get(param_name)
        self = copy.deepcopy(self0)
        
        for i, value in enumerate(values):
            self.set(param_name, value)
            self.calculate_all()
            length_driver_linac[i] = self.length_driver_linacs
            length_plasma_linac[i] = self.length_plasma_linacs
            length_electron_bds[i] = self.linac1.bds.length
            length_positron_bds[i] = self.linac2.bds.length
            length_positron_linac[i] = self.length_main_rf_linacs

        # reset values
        self.set(param_name, orig_value)
        self.calculate_all()

        # total length
        length_total = length_driver_linac+length_plasma_linac+length_electron_bds+length_positron_bds+length_positron_linac

        # plot everything
        fig, ax = plt.subplots(1,1)
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate((length_driver_linac/1e3, np.zeros_like(values))), 
                 label='Driver RF linac')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate((length_driver_linac/1e3, np.flip(length_driver_linac+length_plasma_linac)/1e3)), 
                 label='Plasma linac')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate(((length_driver_linac+length_plasma_linac)/1e3, np.flip(length_driver_linac+length_plasma_linac+length_electron_bds)/1e3)), 
                 label='Electron BDS')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate(((length_driver_linac+length_plasma_linac+length_electron_bds)/1e3, np.flip(length_driver_linac+length_plasma_linac+length_electron_bds+length_positron_bds)/1e3)), 
                 label='Positron BDS')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate(((length_driver_linac+length_plasma_linac+length_electron_bds+length_positron_bds)/1e3, np.flip(length_driver_linac+length_plasma_linac+length_electron_bds+length_positron_bds+length_positron_linac)/1e3)), 
                 label='Positron RF linac')
        ax.plot(values/scale, length_total/1e3, 'k-', label='Collider length')
        ax.plot([orig_value/scale, orig_value/scale], [0.0, np.max(length_total/1e3)*1.1], 'k:')
        
        plt.xscale('log')
        if label is None:
            label = param_name
        if unit is not None:
            label = label + ' (' + unit + ')'
        plt.xlabel(label)
        plt.ylabel('Length (km)')
        plt.ylim(0, np.max(length_total/1e3)*1.1)
        plt.xlim(np.min(values)/scale, np.max(values)/scale)
        plt.legend(loc='upper left', reverse=True)

    # emissions variation plots
    def plot_emissions_variation(self0, param_name, min=None, max=None, unit='a.u.', scale=1, label=None):

        values = np.logspace(np.log10(min), np.log10(max), 100)
        
        emissions_construction = np.empty_like(values)
        emissions_power = np.empty_like(values)
        
        # get original value
        orig_value = self0.get(param_name)
        self = copy.deepcopy(self0)
        
        for i, value in enumerate(values):
            self.set(param_name, value)
            self.calculate_all()
            emissions_construction[i] = self.GWP_tunnels
            emissions_power[i] = self.GWP_power

        # reset values
        self.set(param_name, orig_value)
        self.calculate_all()

        # total cost
        emissions_total = emissions_construction + emissions_power

        # plot everything
        fig, ax = plt.subplots(1,1)
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate((emissions_construction/1e3, np.zeros_like(values))), 
                 label='Construction')
        ax.fill(np.concatenate((values/scale, np.flip(values/scale))), 
                 np.concatenate((emissions_construction/1e3, np.flip(emissions_total)/1e3)), 
                 label='Power')
        ax.plot(values/scale, emissions_total/1e3, 'k-', label='Total emissions')
        ax.plot([orig_value/scale, orig_value/scale], [0.0, np.max(emissions_total/1e3)*1.1], 'k:')
        plt.xscale('log')
        if label is None:
            label = param_name
        plt.xlabel(label + ' (' + unit + ')')
        plt.ylabel('Global Warming Potential (kton CO2e)')
        plt.ylim(0, np.max(emissions_total/1e3)*1.1)
        plt.xlim(np.min(values)/scale, np.max(values)/scale)
        plt.legend(reverse=True)


    ## LAYOUT PLOTS
    
    def plot_layout(self):

        # colormap
        D = np.array([[20, 40, 110],
                      [36, 60, 148],
                      [36, 93, 162],
                      [36, 150, 185],
                      [62, 184, 199],
                      [127, 206, 187],
                      [202, 232, 182],
                      [237, 239, 140],
                      [255, 230, 110],
                      [255, 190, 120]])/255
        F = np.array([0.08, 0.17, 0.25, 0.39, 0.48, 0.56, 0.68, 0.78, 0.86, 1])
        C = np.vstack([np.interp(np.linspace(0, 1, 1000), F, D[:, i]) for i in range(3)]).T
        cmap = matplotlib.colors.ListedColormap(C)

        # prepare figure
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(12)
        col_bds = cmap(0.7)
        col_driver = cmap(0.5)
        col_plasma = cmap(0.99)
        col_rf = cmap(0.2)
        col_turnaround = cmap(0.1)

        # define IP as center
        s_IP = 0
        
        # add arms separately
        for i, linac in enumerate([self.linac1, self.linac2]):

            # plus for arm 1, minus for arm 2
            sign = -(2*i-1)
            
            # add BDS
            s_end_BDS = s_IP
            s_start_BDS = s_end_BDS + sign*linac.bds.length
            ax.plot([s_start_BDS, s_end_BDS], [0,0], color=col_bds, label='BDS {:.0f}'.format(i+1))

            # add linacs
            if linac.pwfa_linac.enable:
                
                # add plasma linac
                s_end_pwfa_linac = s_start_BDS
                s_start_pwfa_linac = s_end_pwfa_linac + sign*linac.pwfa_linac.length
                ax.plot([s_start_pwfa_linac, s_end_pwfa_linac], [0,0], color=col_plasma, label='PWFA linac {:.0f}'.format(i+1))

                # add driver linac
                xs_driver_linac = [0,0]
                if self.use_driver_turnaround:
                    
                    # draw driver turnaround
                    thetas1 = np.linspace(0, 3/2*np.pi, 100)
                    thetas2 = np.linspace(0, np.pi/2, 25)
                    if self.use_driver_turnaround:
                        xs_driver_linac = [10,10]
                        ss1 = s_start_pwfa_linac + self.turn_around.bend_radius*np.sin(thetas1)
                        xs1 = self.turn_around.bend_radius*(1-np.cos(thetas1))
                        ss2 = ss1[-1] + self.turn_around.bend_radius*(np.cos(thetas2)-1)
                        xs2 = xs_driver_linac[1] + xs1[-1] - self.turn_around.bend_radius*np.sin(thetas2)
                        ax.plot(np.concatenate([ss1, ss2]), np.concatenate([xs1, xs2]), label='Turn-around 1', color=col_turnaround)
            
                        # define driver linac position (attach to end of turnaround)
                        s_end_driver_linac = ss2[-1]
                        s_start_driver_linac = s_end_driver_linac - linac.pwfa_linac.driver_linac.length
                        
                else:
                    s_end_driver_linac = s_start_pwfa_linac
                    s_start_driver_linac = s_end_driver_linac + sign*linac.pwfa_linac.driver_linac.length
                    
                # draw driver linac
                ax.plot([s_start_driver_linac, s_end_driver_linac], xs_driver_linac, color=col_driver, label='Driver RF linac {:.0f}'.format(i+1))

                # add PWFA kicker tree
                for i in range(linac.pwfa_linac.num_stages):
                    s0_stage = linac.pwfa_linac.kicker_tree.ss_start_stage[i]
                    ds_stage = linac.pwfa_linac.kicker_tree.ss_end_stage[i]-s0_stage
                    ss = s_start_pwfa_linac - sign*(s0_stage + ds_stage*np.array([0, 1/2, 1]))
                    xs = [0, linac.pwfa_linac.kicker_tree.dxs_stage[i], 0]
                    ax.plot(ss, xs, color='gray', linewidth=0.2)
                
                # TODO: add injector

                # start point of arm
                s_linac_start = s_start_pwfa_linac
                
            else:

                # add main RF linac
                if not self.combined_driver_positron_linac:
                    
                    s_end_rf_linac = s_start_BDS
                    s_start_rf_linac = s_end_rf_linac + sign*linac.rf_linac.length
                    ax.plot([s_end_rf_linac, s_start_rf_linac], [0,0], color=col_rf, label='Main RF linac {:.0f}'.format(i+1))
    
                    # start point of arm
                    s_linac_start = s_start_rf_linac
            
            # add damping ring
            if self.combined_driver_positron_linac:
                loc_damping_ring = (s_start_driver_linac, linac.damping_ring.bend_radius+xs_driver_linac[1])
            else:
                loc_damping_ring = (s_linac_start, linac.damping_ring.bend_radius)
            
            ax.set_aspect('equal', adjustable='box')
            damping_ring = plt.Circle(loc_damping_ring, linac.damping_ring.bend_radius, fill=False)
            ax.add_patch(damping_ring)

        
        # draw positron final turnaround
        if self.combined_driver_positron_linac:
            ss3 = s_start_BDS - self.turn_around.bend_radius*np.sin(thetas1)
            xs3 = self.turn_around.bend_radius*(np.cos(thetas1)-1)
            ss4 = ss3[-1] - self.turn_around.bend_radius*(np.cos(thetas2)-1)
            xs4 = xs3[-1] + self.turn_around.bend_radius*np.sin(thetas2)
            ax.plot(np.concatenate([ss3, ss4]), np.concatenate([xs3, xs4]), label='Turn-around/BDS 2', color=col_turnaround)
            #ax.plot([s_start_driver_linac, s_end_driver_linac], xs_driver_linac, label='Turn-around/BDS 2', color=col_bds)

        # add IP        
        ax.plot([s_IP], [0], 'k*', label='IP')

        # limits and legends
        ax.set_ylim(-0.14*self.length_end_to_end, 0.20*self.length_end_to_end)
        ax.set_xlabel('Length (m)')
        ax.legend(loc='upper left', mode='expand', ncol=4, reverse=True)

