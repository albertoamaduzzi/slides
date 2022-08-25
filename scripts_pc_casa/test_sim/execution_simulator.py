#! /usr/bin/env python3
import sys
import os
import json
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')   

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    from simulator_script import *
    from  sim_objects import *
    from analyzer_script import * 
    from classification_and_plotting import *
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e



try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

def first_configuration_sim_file():
  '''This function initializes the first parameters of the simulation.
  Each epoch a new file be created containing these informations such that the 
  simulation can be initialized for successive steps.
  This file is contained in os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files')
  '''
  enter_parameters_simulation = dict.fromkeys(['config','config0','file_distances_real_data','dir_data','state_basename','start_date',
  'average_fluxes','attraction_activate','locals_','local_distribution','list_new_source','list_reset_source','list_change_source',
  'list_new_attractions','list_reset_attractions','list_change_attractions'])
  enter_parameters_simulation['config'] =os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files') +'\\conf_venezia.json'
  enter_parameters_simulation['config0'] = os.path.join(os.environ['WORKSPACE'],'slides','pvt','conf') +'\\conf.json.local.albi.make'
  enter_parameters_simulation['file_distances_real_data'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data') +'\\COVE flussi_pedonali 18-27 luglio.xlsx'
  enter_parameters_simulation['dir_data'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data') +'\\barriers_config.csv'
  enter_parameters_simulation['state_basename'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides')
  enter_parameters_simulation['start_date'] = '2021-07-14 23:00:00'
  enter_parameters_simulation['stop_date'] = '2021-07-21 23:00:00'
  enter_parameters_simulation['average_fluxes'] = True
  enter_parameters_simulation['attraction_activate'] = True 
  enter_parameters_simulation['locals_'] = True 
  enter_parameters_simulation['local_distribution'] = 'none'  
  enter_parameters_simulation['list_new_source'] =  ['Scalzi_2_IN','Scalzi_3_IN']
  enter_parameters_simulation['list_reset_source'] = ['Costituzione_IN','Papadopoli_IN','Schiavoni_IN','LOCALS']
  enter_parameters_simulation['list_change_source'] = ['Papadopoli_IN']
  enter_parameters_simulation['list_new_attractions'] = ['Farsetti_1_OUT']
  enter_parameters_simulation['list_reset_attractions'] = []
  enter_parameters_simulation['list_change_attractions'] = []

  with open(os.path.join('slides','work_slides','conf_files')+'enter_parameters_simulation.json','w') as f:
      json.dump(enter_parameters_simulation,f)
  return enter_parameters_simulation



if __name__ == '__main__':





    # initialize simulation parameters
    sim = simulator()
    sim.pick_day_input()
    sim.averaging_fluxes()
    sim.normalize_fluxes()
    # copying the objects from metadata
    ch = configuration_handler(list_new_source = sim.list_new_sources,list_reset_source = sim.list_reset_sources,list_change_source = sim.list_change_sources,list_new_attractions = sim.list_new_attractions,list_reset_attractions = sim.list_reset_attractions,list_change_attractions = sim.list_change_attractions, simcfgorig = sim.simcfgorig)    
    # HANDLE SOURCES 
    ch.assign_sources_json(sim.simcfgorig)
    ch.assign_new_sources(sim.list_new_sources, sim.data_barriers,sim.df_avg,sim.simcfgorig)
    ch.reset_sources(sim.list_reset_sources,sim.data_barriers,sim.simcfgorig)
    sim.simcfgorig = ch.assign_sources_to_simcfgorig(sim.simcfgorig)
    # HANDLE ATTRACTIONS 
    ch.assign_attractions_json(sim.simcfgorig)
    ch.assign_new_attractions(sim.list_new_attractions,sim.data_barriers)
    ch.reset_attractions(sim.list_reset_attractions,sim.list_reset_attractions,sim.simcfgorig)
    ch.change_attractions(sim.list_change_attractions,sim.simcfgorig)
    sim.simcfgorig = ch.assign_attractions_to_simcfgorig(sim.simcfgorig,sim)
    # DEFINE OUTPUT
    sim.assign_directory_state_basename(ch.dict_sources)
    # the state_basename is being defined.
    # run the simulation
    sim.run_sim()

#    tnow = datetime.now()
#    with Pool(processes=sim.nagents) as pool:
#      result = pool.map(sim.run_simulation, sim.list_delta_u_attraction, sim.chunksize)
#      tscan = datetime.now() - tnow
#      print(f'Scan took {tscan}')
    # ANALISYS
    analisys = analyzer(sim)
    analisys.produce_comparison_df(sim)
    analisys.distance_csv_for_ward(sim)
    analisys.correlation_matrix_plot(sim)    
    analisys.ward_plot(sim)
    # clustering procedure
    
    
    # plotting procedure


    '''
    for n in n_epochs:
      if n == 0:
        enter_parameters_simulation = first_configuration_sim_file()
      else:
        enter_parameters_simulation = updated_configuration_sim_file
      main(enter_parameters_simulation)
    '''
    