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
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')   

#try:
sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
from simulator_script import simulator
from sim_objects import configuration_handler,attraction,source
from analyzer_script import analyzer,barrier  
from classification_and_plotting import classifier,plotter
#except Exception as e:
#  raise Exception('library loading error : {}'.format(e)) from e



try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

def first_configuration_sim_file(list_new_sources,list_days):
  '''This function initializes the first parameters of the simulation.
  Each epoch a new file be created containing these informations such that the 
  simulation can be initialized for successive steps.
  This file is contained in os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files')
  list_new_sources and list_days must always be lists of lists [[set_sources_0],..,[set_sources_n]] ... for days too.
  '''
  for s_ in list_new_sources:
    c=0
    for d_ in list_days:
      print(s_)
      print(d_)
      enter_parameters_simulation = dict.fromkeys(['config','config0','file_distances_real_data','dir_data','state_basename','start_date',
      'average_fluxes','attraction_activate','locals_','local_distribution','list_new_source','list_reset_source','list_change_source',
      'list_new_attractions','list_reset_attractions','list_change_attractions'])
      enter_parameters_simulation['config'] =os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','conf_venezia.json') 
      enter_parameters_simulation['config0'] = os.path.join(os.environ['WORKSPACE'],'slides','pvt','conf','conf.json.local.albi.make') 
      enter_parameters_simulation['file_distances_real_data'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','COVE flussi_pedonali 18-27 luglio.xlsx')
      enter_parameters_simulation['dir_data'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','barriers_config.csv') 
      enter_parameters_simulation['state_basename'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides')
      enter_parameters_simulation['file_cnt'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','venice.fluxes')
      enter_parameters_simulation['start_date'] = d_[0]#'2021-07-14 23:00:00'
      enter_parameters_simulation['stop_date'] = d_[1]#'2021-07-15 22:59:00'
      enter_parameters_simulation['average_fluxes'] = False
      enter_parameters_simulation['attraction_activate'] = True 
      enter_parameters_simulation['local_distribution'] = 'none'  
      enter_parameters_simulation['list_new_source'] =  s_ #['Scalzi_2_IN','Scalzi_3_IN','Papadopoli_1_IN']
      enter_parameters_simulation['list_reset_source'] = ['LOCALS'] #[] # 'Costituzione_IN','Papadopoli_IN','Schiavoni_IN', # 13165  16479  14592  12331  10369  10058  11486  17261  34596  58961 83676 116294 153646 189855 221815 254315 287078 315350 344479 371515 393474 404686 410937 414293
      if len(enter_parameters_simulation['list_reset_source']) == 0:
        enter_parameters_simulation['locals_'] = False
      else: 
        enter_parameters_simulation['locals_'] = True 
      enter_parameters_simulation['list_change_source'] = []
      enter_parameters_simulation['list_new_attractions'] = ['Maddalena_1_IN']
      enter_parameters_simulation['list_reset_attractions'] = []
      enter_parameters_simulation['list_change_attractions'] = []
      string_name = ''
      for e in s_:
        string_name = string_name + e 
      with open(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','enter_parameters_simulation_{0}_{1}.json'.format(string_name,c)),'w') as f:
          json.dump(enter_parameters_simulation,f,indent = 4)
      c = c + 1
  return enter_parameters_simulation

def main(enter_parameters_simulation):
  print('just entered main, list_new_source:\t',enter_parameters_simulation['list_new_source'])
  list_cost_corr = []
  list_symmetric_ward_distance_sim_real = []
  list_symmetric_ward_distance_sim_real_norm = []
  list_ward_distance_sim_real = []
  list_ward_distance_sim_real_norm = []
  for n_epoch in range(1):
    if n_epoch == 0:
      output_sim,dict_dir,plots_,map_,dendogram_ = '','','','',''                
    # initialize simulation parameters
      sim = simulator(enter_parameters_simulation)
      sim.pick_day_input()
      sim.averaging_fluxes()
      sim.normalize_fluxes()
    # copying the objects from metadata
    ch = configuration_handler(n_epoch = n_epoch,list_new_source = sim.list_new_sources,list_reset_source = sim.list_reset_sources,list_change_source = sim.list_change_sources,list_new_attractions = sim.list_new_attractions,list_reset_attractions = sim.list_reset_attractions,list_change_attractions = sim.list_change_attractions, simcfgorig = sim.simcfgorig)    
    # HANDLE SOURCES 
    ch.assign_sources_json(sim.simcfgorig)
    if sim.average_fluxes:
      ch.assign_new_sources(sim.list_new_sources, sim.data_barriers,sim.df_avg,sim.simcfgorig)
    else:
      ch.assign_new_sources(sim.list_new_sources, sim.data_barriers,sim.df_day,sim.simcfgorig)      
    ch.reset_sources(sim.list_reset_sources,sim.data_barriers,sim.simcfgorig)
    sim.simcfgorig = ch.assign_sources_to_simcfgorig(sim.simcfgorig)
#    print('sources of simcfgorig',sim.simcfgorig['sources'])
#    print('source file conf scalzi 2:\n',sim.df_day['SCALZI_2_IN'],'\nscalzi3\n',sim.df_day['SCALZI_3_IN'])
    # HANDLE ATTRACTIONS 
    ch.assign_attractions_json(sim.simcfgorig)
    ch.assign_new_attractions(sim.list_new_attractions,sim.data_barriers)
    ch.reset_attractions(sim.list_reset_attractions,sim.list_reset_attractions,sim.simcfgorig)
    ch.change_attractions(sim.list_change_attractions,sim.simcfgorig)
    sim.simcfgorig,output_sim,dict_dir = ch.assign_attractions_to_simcfgorig(sim,output_sim,dict_dir,n_epoch)
    # DEFINE OUTPUT
    run_ = True
#    print('configuration file I am using',sim.simcfgorig)
    with open(os.path.join(os.environ['WORKSPACE'],'minimocas-tools','work_lavoro','file_conf.json'),'w') as f:
      json.dump(sim.simcfgorig,f,indent = 4)
    if run_:
      sim.run_sim()
    # ANALISYS
    print('before analisys',sim.state_basename)
    analisys = analyzer(sim,n_epoch)
    analisys.create_dict_barrier_sim_real(sim)
    
#    analisys.produce_comparison_df(sim) # in alternative to the line before
    analisys.distance_csv_for_ward(sim,dict_dir)
    plots_, cost_corr = analisys.correlation_matrix_plot(sim,plots_)
    analisys.comparison_number_people_sim_real(sim,plots_)  
    comparison_by_plots = True
    if comparison_by_plots:
      analisys.focus_comparison_fluxes(sim,plots_)
      analisys.comparison_corr_subsets(sim,plots_)
    compare_cumulative = False
    if compare_cumulative:
      analisys.compare_cumulative_fluxes(sim) 
    pla = True
    if pla:
      dendogram_,symmetric_ward_distance_sim_real,symmetric_ward_distance_sim_real_norm,ward_distance_sim_real,ward_distance_sim_real_norm = analisys.ward_plot(sim,dendogram_)    
      if 1==0:
        analisys.plot_cluster_real_behavior(sim,plots_)  
    # plotting procedure
      list_symmetric_ward_distance_sim_real.append(symmetric_ward_distance_sim_real)
      list_symmetric_ward_distance_sim_real_norm.append(symmetric_ward_distance_sim_real_norm)
      list_ward_distance_sim_real.append(ward_distance_sim_real)
      list_ward_distance_sim_real_norm.append(ward_distance_sim_real_norm)
      print('ward distance',symmetric_ward_distance_sim_real)
      # PLOTTER
      plott = plotter(sim,analisys,n_epoch)
      map_ = plott.common_map_in(sim,ch,map_)
      map_ = plott.common_map_out(sim,ch,map_)
      map_ = plott.common_map_in_circle(sim,ch,map_)
#      mappa,map_ = plott.common_map(sim,ch,map_)
      yes = True
      if yes:
#       plott.map_with_plots(sim,ch,map_,analisys)
        plott.map_with_plots_in(sim,ch,map_,analisys)
        plott.map_with_plots_out(sim,ch,map_,analisys)
        plott.map_best_worst_euclidean_temporal_distance_in(sim,ch,map_,analisys)
        plott.map_best_worst_euclidean_temporal_distance_out(sim,ch,map_,analisys)
        plott.map_ward_clustering_in(sim,analisys,ch,map_)      
        plott.map_ward_clustering_out(sim,analisys,ch,map_)      
      return list_cost_corr,list_symmetric_ward_distance_sim_real,list_symmetric_ward_distance_sim_real_norm,list_ward_distance_sim_real,list_ward_distance_sim_real_norm
    else:
      list_cost_corr.append(cost_corr)
      plott = plotter(sim,analisys,n_epoch)
      map_ = plott.common_map_in(sim,ch,map_)
      map_ = plott.common_map_out(sim,ch,map_)
      plott.map_good_correlated_in(analisys,sim,ch,map_)
      plott.map_good_correlated_out(analisys,sim,ch,map_)
#      mappa,map_ = plott.common_map(sim,ch,map_)
      yes = True
      if yes:
        plott.map_with_plots_in(sim,ch,map_,analisys)                        
        plott.map_with_plots_out(sim,ch,map_,analisys)            
        plott.map_with_plots(sim,ch,map_,analisys)
        plott.map_best_worst_euclidean_temporal_distance_in(sim,ch,map_,analisys)
        plott.map_best_worst_euclidean_temporal_distance_out(sim,ch,map_,analisys)
      return list_cost_corr
if __name__ == '__main__':
  parallel_ = False
  if parallel_:
    list_enter_parameters_simulation = []
    #list_new_sources = [["Scalzi_2_IN"],["Scalzi_2_IN","Scalzi_3_IN"],["Scalzi_2_IN","Scalzi_3_IN","Papadopoli_1_IN"],["Scalzi_2_IN","Scalzi_3_IN","Costituzione_1_IN"],["Scalzi_2_IN","Scalzi_3_IN","Costituzione_1_IN","Papadopoli_1_IN"],
    #                                                     ["Scalzi_2_IN","Papadopoli_1_IN"],["Scalzi_2_IN","Costituzione_1_IN"],["Scalzi_2_IN","Papadopoli_1_IN","Costituzione_1_IN"],
    #                                                     ["Scalzi_3_IN"],
    #                                                     ["Scalzi_3_IN","Papadopoli_1_IN"],["Scalzi_3_IN","Costituzione_1_IN"],["Scalzi_3_IN","Papadopoli_1_IN","Costituzione_1_IN"],
    #                                                     ["Papadopoli_1_IN"],
    #                                                     ["Papadopoli_1_IN","Costituzione_1_IN"],
    #                                                     ["Costituzione_1_IN"]]
    list_new_sources = [["Scalzi_2_IN","Scalzi_3_IN"],["Scalzi_2_IN"],["Scalzi_3_IN"]]
    list_days = [['2021-07-12 23:00:00','2021-07-13 22:59:00'],
                  ['2021-07-13 23:00:00','2021-07-14 22:59:00'],['2021-07-14 23:00:00','2021-07-15 22:59:00'],
                  ['2021-07-15 23:00:00','2021-07-16 22:59:00'],['2021-07-16 23:00:00','2021-07-17 22:59:00'],
                  ['2021-07-17 23:00:00','2021-07-18 22:59:00']]
    first_configuration_sim_file(list_new_sources,list_days)
    if 1==1:
      for j in range(len(list_days)):
        for s_ in list_new_sources:
          string_name = ''
          for e in s_:
            string_name = string_name + e
          print('name sources simulation\t',string_name) 
          with open(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','enter_parameters_simulation_{0}_{1}.json'.format(string_name,j)),'r') as f:
            enter_parameter_simulation = json.load(f)
          list_enter_parameters_simulation.append(enter_parameter_simulation)
      with Pool(processes=3) as pool:
        result = pool.map(main,list_enter_parameters_simulation)
  else:
    list_new_sources = [['Papadopoli_1_IN','Costituzione_1_IN','Schiavoni_1_IN']]#[['Costituzione_1_IN']]# ['Piazzale_Roma_IN','Papadopoli_1_IN','Costituzione_1_IN'],#[["Scalzi_2_IN","Scalzi_3_IN"],["Scalzi_2_IN"],["Scalzi_3_IN"],["Scalzi_2_IN","Scalzi_3_IN","Papadopoli_1_IN"]] # ,"Costituzione_1_IN" #[['Piazzale_Roma_IN']]  
    list_days = [['2021-07-12 23:00:00','2021-07-13 22:59:00']]
#                 ,                  ['2021-07-13 23:00:00','2021-07-14 22:59:00'],['2021-07-14 23:00:00','2021-07-15 22:59:00'],
#                  ['2021-07-15 23:00:00','2021-07-16 22:59:00'],['2021-07-16 23:00:00','2021-07-17 22:59:00'],
#                  ['2021-07-17 23:00:00','2021-07-18 22:59:00']]
    enter_parameter_simulation = first_configuration_sim_file(list_new_sources,list_days)
    for j in range(len(list_days)):
      print('day simulation number:\t',j)
      for s_ in list_new_sources:
        string_name = ''
        for e in s_:
          string_name = string_name + e
        print('name sources simulation:\t',string_name)         
        with open(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','enter_parameters_simulation_{0}_{1}.json'.format(string_name,j)),'r') as f:
          enter_parameter_simulation = json.load(f)
#        print('starting day simulation',enter_parameter_simulation['start_date'],'day number',j)
#        print('reading','enter_parameters_simulation_{0}_{1}.json'.format(string_name,j))
        main(enter_parameter_simulation)
         
    '''
    for n in n_epochs:
      if n == 0:
        enter_parameters_simulation = first_configuration_sim_file()
      else:
        enter_parameters_simulation = updated_configuration_sim_file
      main(enter_parameters_simulation)
    '''
