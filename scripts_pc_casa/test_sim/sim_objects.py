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
import seaborn as sns
warnings.filterwarnings('ignore')  
#### THESE CLASSES ARE USED FOR THE PREPROCESSING TO BE GIVEN TO THE SIMULATOR ####
try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    from simulator_script import *
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e


class configuration_handler:
    '''Configuration handler class:
    ---------------
    Input:
    dict_sources: dict -> ['Pappadopoli_IN','Costituzione_IN']
    dict_attractions: dict -> ['Farsetti_1_OUT']
    list_modified_attractions: list ->
    list_delta_u_attractions: list -> [['Farsetti_1',0.05],['Farsetti_1',-0.05],...] (list.shape() = (len(list_modified_attractions),1))
    list_added_sources: list -> ['Scalzi_1_IN,Scalzi_2_IN']
    list_resetted_sources: list -> ['Schiavoni_IN']
    config: str -> /path/to/conf_venezia.json
    config0: str -> /path/to/conf.json.local.albi.make
    simcfgorig: file_json -> conf_venezia.json
    '''
    def __init__(self,
                n_epoch,
                list_new_source,
                list_reset_source,
                list_change_source,
                list_new_attractions,
                list_reset_attractions,
                list_change_attractions,
                simcfgorig,
                dict_sources = {},
                dict_attractions = {},
                list_delta_u_attractions = [],
                ): 
        # DICT SOURCES
        number_sources = ['1','2','3','4']
        self.list_sources_simcfgorig = []
        for s_name in list(simcfgorig['sources'].keys()):
            if any([n not  in s_name for n in number_sources]) and s_name != 'LOCALS':
                self.list_sources_simcfgorig.append(s_name.split('_')[0] + '_1_' + s_name.split('_')[1])
            else:
                self.list_sources_simcfgorig.append(s_name)
#        self.dict_sources = dict.fromkeys(list(np.concatenate((list_new_source,list_reset_source,list_change_source,self.list_sources_simcfgorig))))
        try:
            self.dict_sources = dict.fromkeys(list(np.concatenate((list_new_source,list_reset_source,self.list_sources_simcfgorig))))
        except ValueError:
            self.dict_sources = dict.fromkeys(list(list_new_source))
        # DICT ATTRACTIONS
        for s_ in list(simcfgorig['sources'].keys()):
            self.dict_sources.pop(s_,None)
        self.dict_attractions = dict.fromkeys(list(np.concatenate((list_new_attractions,list_reset_attractions,list_change_attractions,list(simcfgorig['attractions'].keys())))))
        
        self.list_delta_u_attractions = list_delta_u_attractions
        self.is_assigned_new_sources = False
        self.is_assigned_default_sources = False
        self.is_resetted_sources = False
        self.windows = True
        self.n_epoch = n_epoch
        ######## HANDLE SOURCES #########
    

    def assign_sources_json(self,simcfgorig):
        '''Input:
    simcfgorig: file_json with all sim infos
    USAGE:
    Initializes dict_sources with sources from json default
    DESCRIPTION:
    mandatory before reset_sources(), assign_sources_to_json()
    '''
        number_sources = ['1','2','3','4']
        for s_name in simcfgorig['sources']:
            s = source()
            if any([n not  in s_name for n in number_sources]) and s_name != 'LOCALS':
                s.name = s_name.split('_')[0] + '_1_' + s_name.split('_')[1]
            else:
                s.name = s_name
            s.is_default = True
            for value in simcfgorig['sources'][s_name]:
                if type(simcfgorig['sources'][s_name][value]) == int:
                    s.creation_dt = simcfgorig['sources'][s_name][value]
                elif type(simcfgorig['sources'][s_name][value]) == list:
                    s.creation_rate = simcfgorig['sources'][s_name][value]
                elif type(simcfgorig['sources'][s_name][value]) == dict:
                    if value == 'source_location':
                        s.source_location = simcfgorig['sources'][s_name][value]
                    else: 
                        s.pawns_from_weight = simcfgorig['sources'][s_name][value]
            self.dict_sources[s.name] = s
        self.is_assigned_default_sources = True
        return True
                 
                 
    def assign_new_sources(self,
                          list_new_source,
                          data_barriers,
                          df_avg,
                          simcfgorig
                          ):
        '''Input:
        list_added_sources: from the simulator
        data_barriers = file_csv containing barriers already opened
        df_avg = file_csv inherited from simulations containing validation data.
                  '''
        for s_ in list_new_source:
            print('assigning new sources:\t',list_new_source)
            if 'Costituzione_1_IN' == s_ and not 'Piazzale_Roma_IN' in list_new_source: # NOTE: this decision is done as papadopoli contributes to costituzione in the sim, and also with scalzi_2,scalzi_3 
                list_sources_to_agglomerate = ["SCALZI_2_IN","SCALZI_3_IN","Costituzione_1_IN"]
                s = source()
                s.name = s_
                s.is_added = True
                s.creation_dt = 30
                s.source_location = {'lat':data_barriers.loc[data_barriers['Description']==s_.upper()].iloc[0]['Lat'],'lon':data_barriers.loc[data_barriers['Description']==s_.upper()].iloc[0]['Lon']}
                s.pawns_from_weight = {
            'tourist':{'beta_bp_miss' : 0,
                'speed_mps': 1}}
                s.creation_rate = []
                list_flux_roma = np.zeros(96)
                for s_ in list_sources_to_agglomerate:
                    c = 0
                    for flux in df_avg[s_.upper()]:
                        for j in range(4):
                            list_flux_roma[c] = list_flux_roma[c] + flux/4 
                            c = c + 1
                            
                s.creation_rate = list(list_flux_roma)
                print('s.creation_rate costituzione',s.creation_rate)                
                if len(s.creation_rate)!= 96:
                    print('la lunghezza della lista per la sorgente {0} è {1}'.format(s.name,len(s.creation_rate)))
                    exit( )
                self.dict_sources[s.name] = s

            elif not 'Piazzale_Roma_IN' == s_:
                print('assigning sources: source\t',s_)
                s = source()
                s.name = s_
                s.is_added = True
                s.creation_dt = 30
                if s_ == 'Schiavoni_1_IN':
                    s.creation_rate = [22.567046242405045,17.138238639159386,17.375692873795348,17.018392585466678,16.1996957691111,14.62937748501904,15.622877143647353,9.408416528479965,11.211719643860704,9.712439597619584,12.594390164178048,11.128288092081423,13.76933342689697,15.60265747676268,12.393624485620805,15.677489717996998,23.497527259245285,20.476383027321035,27.993698071742,31.35989023199272,33.7102351662263,48.040141379039696,53.72493964473563,44.544435265203354,72.44298298894365,64.6942200320617,86.49318288805077,91.74543314906906,101.6955741362462,101.1887524060395,95.9556512551255,122.92755072769285,130.8053020722569,113.10406564595918,120.19174931929774,116.46904136815164,124.35735959856409,134.52328714077908,106.86637449866716,135.08121726029643,118.10019513202614,107.93047228510738,111.20184189009244,114.15194763213678,94.02850471064964,94.28737675930063,98.77081955325602,93.04914051743982,111.38265728430908,92.79109239044665,87.06010827132188,79.83235476401494,80.8065638537392,102.82248457401325,107.19520123000879,117.35367097314239,112.52954045596756,104.4986330254436,116.30069041350178,137.8923447313935,136.54518938518615,129.0107632099219,135.2915438247804,122.79725548782682,119.23726024422552,118.53297490045412,139.9850980263819,115.14729654054806,111.79320260920161,120.49389578580944,125.87907529087362,118.6374413287923,107.44569986172839,109.88063925277136,112.94847174113106,98.74456307926333,98.22278370520722,103.94641423424324,96.27364945053532,80.17440037620416,101.63251810680582,75.04404104507152,80.82527372323783,71.35894426424534,67.19260320439516,63.578157050985105,59.332225353020405,59.351997173547176,57.266407349383236,49.64666865896931,45.88463651377644,41.647939604931985,32.62558146257313,28.663056246242306,24.8645088643431,24.173508191266148]
                    s.source_location ={"lat": 45.433819,"lon": 12.344299}
                    s.pawns_from_weight = {"tourist": {
                    "beta_bp_miss": 0.5,
                    "speed_mps": 1.0}}
                else:
                    s.source_location = {'lat':data_barriers.loc[data_barriers['Description']==s_.upper()].iloc[0]['Lat'],'lon':data_barriers.loc[data_barriers['Description']==s_.upper()].iloc[0]['Lon']}
                    s.pawns_from_weight = {
                    'tourist':{'beta_bp_miss' : 0,
                    'speed_mps': 1}}
                    s.creation_rate = []
                    for flux in df_avg[s_.upper()]:
                        for j in range(4):
                            s.creation_rate.append(flux/4)
                    if len(s.creation_rate)!= 96:
                        print('la lunghezza della lista per la sorgente {0} è {1}'.format(s.name,len(s.creation_rate)))
                        exit( )
                self.dict_sources[s.name] = s
                
            elif 'Piazzale_Roma_IN' == s_:
                list_sources_to_agglomerate = ["SCALZI_2_IN","SCALZI_3_IN"]
                s = source()
                s.name = 'Piazzale_Roma_IN'
                s.is_added = True
                s.creation_dt = 30
                s.source_location = {'lat':45.440817,'lon':12.321442}
                s.pawns_from_weight = {
            'tourist':{'beta_bp_miss' : 0,
                'speed_mps': 1}}
                s.creation_rate = []
                list_flux_roma = np.zeros(96)
                for s_ in list_sources_to_agglomerate:
                    c = 0
                    for flux in df_avg[s_]:
                        for j in range(4):
                            list_flux_roma[c] = list_flux_roma[c] + flux/4 
                            c = c + 1
                            
                s.creation_rate = list(list_flux_roma)                
                if len(s.creation_rate)!= 96:
                    print('la lunghezza della lista per la sorgente {0} è {1}'.format(s.name,len(s.creation_rate)))
                    exit( )
                self.dict_sources[s.name] = s
            else:
                pass
        self.is_assigned_new_sources = True
            
        return True 
    
    
    def show_name_sources(self):
        for s in self.dict_sources.values():
            print(s.name)
    def show_name_attractions(self):
        for a in self.dict_attractions.values():
            print(a.name)
        
    
    def reset_sources(self,list_reset_source,data_barriers,simcfgorig):
        '''Input: 
        reset_source_list: from the simulator
        data_barriers = file_csv containing barriers already opened
        '''
        number_source = ['1','2','3','4']
        if 'LOCALS' in list_reset_source:
            self.locals_ = True
        for s_ in list_reset_source:
            if s_ == 'LOCALS':
                s = source()
                s.name = s_
                s.is_reset = True
                s.is_added = False
                s.is_default = False
                s.is_changed = False
                s.creation_rate = list(np.zeros(96))
                s.pawns = {"locals": {
          "beta_bp_miss": 0,
          "start_node_lid": -1,
          "dest": -1}}
                self.dict_sources[s.name] = s
            else:
                print('this time there are locals')
                pass
#            self.dict_sources[s.name] = s

            '''
            s = source()
            if any([n not in s_ for n in number_source]):
                s.name = s_.split('_')[0] + '_' + number_source[0] +'_' + s_.split('_')[1]
            else:
                s.name = s_
            s.is_reset = True
            s.is_added = False
            s.is_default = False
            s.is_changed = False
            try:
                s.pawns_from_weight = {'tourist':{'beta_bp_miss' : simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['beta_bp_miss'],
                'speed_mps': simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['speed_mps']}}
                s.source_location = {'lat':data_barriers.loc[data_barriers['Description']==s.name.upper()].iloc[0]['Lat'],'lon':data_barriers.loc[data_barriers['Description']==s.name.upper()].iloc[0]['Lon']}
            except IndexError:
                print('source {} is not found in the barriers'.format(s_))
    self.dict_sources[s.name] = s
    self.is_resetted_sources = True'''

    
     

    def assign_sources_to_simcfgorig(self,simcfgorig):
        '''Input: 
        simcfgorig: json_file
        Description: for each source in dict_sources adds all the infos. Is called when all the sources are initialized'''
        print('dictionary sorces',self.dict_sources['LOCALS'].__dict__)
        for s in self.dict_sources.values():
            if s.is_added == True or s.is_changed == True:
                simcfgorig['sources'][s.name ] = {'creation_dt' : 30,'creation_rate' : s.creation_rate,
        'source_location' : {'lat' : s.source_location['lat'],'lon' : s.source_location['lon']},
        'pawns_from_weight': s.pawns_from_weight}
            elif s.is_reset == True and s.name!='LOCALS':
                simcfgorig['sources'][s.name ] = {'creation_dt' : 30,
                                                  'creation_rate' : np.zeros(96).tolist(),
                                                  'source_location' : {'lat' : s.source_location['lat'],
                                                                       'lon' : s.source_location['lon']},
                                                                       'pawns_from_weight': s.pawns_from_weight}                
            else:
                simcfgorig['sources'][s.name] = {"source_type": s.source_type,
                                                 "creation_dt": 30,
                                                 "creation_rate": s.creation_rate,
                                                 "pawns": s.pawns_from_weight}
                print(s.name,s.creation_rate,s.is_reset)
        return simcfgorig
                

        ###### HANDLE ATTRACTIONS #############

    def assign_attractions_json(self,simcfgorig):
        '''Input:
        simcfgorig: file_json with all sim infos
        USAGE:
        Initializes dict_attractions with attractions from json default
        DESCRIPTION:
        mandatory before reset_attractions(), assign_attractions_to_json()
        '''
        for a_name in simcfgorig['attractions']:
            a = attraction()
            a.name = a_name
            a.is_default = True
            for value in simcfgorig['attractions'][a.name]:
                if type(simcfgorig['attractions'][a.name][value]) == float:
                    if value == 'lat':
                        a.lat = simcfgorig['attractions'][a.name][value]
                    elif value == 'lon':
                        a.lon = simcfgorig['attractions'][a.name][value]
                    elif value == 'visit_time':
                        # PUT VISIT TIME BY HAND
                        if a.name == 'Campo_Santa_Margherita':
                            a.visit_time = 20000
                        else:    
                            a.visit_time = simcfgorig['attractions'][a.name][value]
                else:
                    if value == 'weight':
                        # INSERT SPECIFIC MODIFICATIONS COMMENT FOR ELIMINATION. DATA SUGGEST IT IS MORE ATTRACTIVE THEN THOUGHT
                        if a.name == 'Campo_Santa_Margherita':
#                            a.weight = list(np.zeros(24))
                            a.weight = list(np.ones(24)*0.6)
#                        elif a.name == 'Basilica_dei_Frari':
#                            a.weight = list(np.zeros(24))
#                            a.weight = list(np.ones(24)*0.3)
                        elif a.name == 'Scuola_Grande_di_San_Rocco':
                            a.weight = list(np.ones(24)*0.1)
#                            a.weight = list(np.zeros(24))
#                        elif a.name == 'Campo_San_Polo':
#                            a.weight = list(np.zeros(24))
#                            a.weight = list(np.ones(24)*0.3)
                        elif a.name == 'Ponte_di_Rialto':
#                            a.weight = list(np.zeros(24))
                            a.weight = list(np.ones(24)*0.9)
                        elif a.name == "Gallerie_dell'Accademia":
                            a.weight = list(np.ones(24)*0.8)
                        elif a.name == "Riva_degli_Schiavoni":
                            a.weight = list(np.ones(24)*0.8)                        
                        elif a.name == 'San_Marco':
                            a.weight = list(np.ones(24))
                            a.weight[:8] = list(np.ones(8)*0.6)
#                            print('San Marco', a.weight)
#                        elif a.name == 'Farsetti_1_IN':
#                            a.weight = list(np.ones(24))                        
                        elif a.name == 'Chiesa_di_San_Zaccaria':
                            a.weight = list(np.ones(24))                        

                        else:
                            a.weight = simcfgorig['attractions'][a.name][value]
#                            a.weight = list(np.zeros(24))
                    else:
                        a.time_cap = simcfgorig['attractions'][a.name][value]
            self.dict_attractions[a.name] = a
        return True

                 
                 
                 
    def assign_new_attractions(self,
                          list_new_attractions,
                          data_barriers,
                          type_ = 'B'
                          ):
        '''Input:
        list_new_attractions: from the simulator
        data_barriers = file_csv containing barriers already opened
        df_avg = file_csv inherited from simulations containing validation data.
        type_ = type of attraction: allowed 'A', 'B'
                  '''
        for a_ in list_new_attractions:
            a = attraction()
            a.name = a_ 
            a.is_added = True
#            if a_.upper() in data_barriers['Description']:
            a.lat = data_barriers.loc[data_barriers['Description']==a_.upper()].iloc[0]['Lat']
            a.lon = data_barriers.loc[data_barriers['Description']==a_.upper()].iloc[0]['Lon']
#            else:
#                print(a_,'is not found in the barrier file')
            if type_ == 'A':
                a.weight = list(np.ones(24)*0.3)
            else:
                a.weight = list(np.ones(24)*0.6)
            a.time_cap = list(np.ones(24)*1000)
            a.visit_time = 3600
            self.dict_attractions[a_] = a
        return True
        
    def reset_attractions(self,list_reset_attractions,data_barriers,simcfgorig):
        '''Input: 
        list_reset_attractions : from the simulator
        data_barriers = file_csv containing barriers already opened
        '''
        for a_ in list_reset_attractions:
            a = attraction()
            a.name = a_
            a.is_reset = True
            a.is_added = False
            a.is_default = False
            a.is_changed = False
            if a_.upper() in data_barriers['Description']:
                a.lat = data_barriers.loc[data_barriers['Description']==a_.upper()].iloc[0]['Lat']
                a.lon = data_barriers.loc[data_barriers['Description']==a_.upper()].iloc[0]['Lon']
            else:
                print(a_,'is not in not found in the barrier file')
            a.weight = list(np.zeros(24))
            a.time_cap = list(np.ones(24)*1000)
            a.visit_time = 280
            self.dict_attractions[a_] = a


            
    def change_attractions(self,list_change_attractions,simcfgorig):
        '''Input:
        list_change_attractions: list -> that contains the list of attractions whose vector of attractivity
        must be changed
        TODO: add a way to decide which attractions to change and how much.'''
        self.delta_u = np.linspace(-0.1,0.1,10)
        self.dict_changed_attractions = dict.fromkeys(list_change_attractions)
        for a_ in list_change_attractions:
            a = attraction()
            a.name = a_
            a.is_reset = False
            a.is_added = False
            a.is_default = False
            a.is_changed = True
            a.weight = simcfgorig['attractions'][a.name]['weight'] + list(np.ones(24)*self.delta_u)
            a.time_cap = list(np.ones(24)*1000)
            a.visit_time = 2880
        for attractions in list_change_attractions:
            self.dict_changed_attractions[a].weight = a.weight
        
    
    def assign_attractions_to_simcfgorig(self,sim_,output_sim_,dict_dir,n_epoch):
        '''Input: 
        simcfgorig: json_file
        Description: for each source in dict_sources adds all the infos. Is called when all the sources are initialized
        Then it returns simcfgorig of th simulation
        NOTE:
        I have added as an argument n_epoch as otherwhise in the iteration I would os.path.join many times the state_basename via sim.assign_dir_basename'''
        json_string = dict.fromkeys(list(self.dict_attractions.keys()))
        # HANDLE SOURCES FOR CONVENIENCE HERE
        json_string_sources = dict.fromkeys(list(self.dict_sources.keys()))
        for s in self.dict_sources.values():
            if s.name != 'LOCALS':
                json_string_sources[s.name] = {'creation_rate':s.creation_rate,'pawns_from_weight':s.pawns_from_weight}
            else:
                json_string_sources[s.name] = {'creation_rate':s.creation_rate,'pawns':s.pawns}
                
        # END HANDLING OF SOURCES HERE
        for a in self.dict_attractions.values():
            if a.is_changed:
                sim_.simcfgorig['attractions'][a.name]['weight'] = a.weight
            else:
                sim_.simcfgorig['attractions'][a.name] = {'lat' : a.lat,'lon' : a.lon, 'weight' : a.weight, 'timecap' : a.time_cap, 'visit_time' : a.visit_time}
      ### CREATION FILE PARAMETERS ATTRACTIONS and SOURCES ###
            json_string[a.name] = {'weight':a.weight,'timecap':a.time_cap,'visit_time':a.visit_time}
        ### HANDLE DIR ###
        if n_epoch == 0:
            sim_.assign_directory_state_basename(self.dict_sources)
        if self.windows:
            if self.n_epoch == 0:
                dict_dir = '/dict_dir_{}'.format(0)
            else:
                dict_dir = dict_dir + '/dict_dir_{}'.format(self.n_epoch)
        else:
            if self.n_epoch == 0:
                dict_dir = '\\dict_dir_{}'.format(0)
            else:
                dict_dir = dict_dir + '\\dict_dir_{}'.format(self.n_epoch)
        if not os.path.exists(sim_.state_basename + dict_dir):
            os.mkdir(sim_.state_basename + dict_dir)
        if self.windows:        
            with open (sim_.state_basename + dict_dir +'/attractions_present_simulation.json','w') as outfile:
                json.dump(json_string,outfile,indent = 4)
            with open (sim_.state_basename + dict_dir +'/sources_present_simulation.json','w') as outfile:
                json.dump(json_string_sources,outfile,indent = 4)

        else:        
            with open (sim_.state_basename + dict_dir +'\\attractions_present_simulation.json','w') as outfile:
                json.dump(json_string,outfile,indent = 4)
            with open (sim_.state_basename + dict_dir +'\\sources_present_simulation.json','w') as outfile:
                json.dump(json_string_sources,outfile,indent = 4)
        
        ### INITIALIZE STATE_BASENAME###
        if self.windows:
            if self.n_epoch == 0:
                output_sim_ = '/output_sim_{}'.format(0)
            else:
                output_sim_ = output_sim_ + '/output_sim_{}'.format(self.n_epoch)
        else:
            if self.n_epoch == 0:
                output_sim_ = '\\output_sim_{}'.format(0)
            else:
                output_sim_ = output_sim_ + '\\output_sim_{}'.format(self.n_epoch)
        if not os.path.exists(sim_.state_basename + output_sim_):
            os.mkdir(sim_.state_basename + output_sim_)
        if self.windows:                
            sim_.simcfgorig['state_basename'] = sim_.state_basename + output_sim_ + '/venezia'
        else:                
            sim_.simcfgorig['state_basename'] = sim_.state_basename + output_sim_ + '\\venezia'
        sim_.path_output_sim = sim_.state_basename + output_sim_
#        print('printing the attractions givento simcfgorig',sim_.simcfgorig['attractions'])
#        print('configuration_handler dict',self.__dict__)
        return sim_.simcfgorig,output_sim_,dict_dir
        


class source:
    '''Source class:
    ----------------
    Input:
    name: str -> Papadopoli_IN
    creation_dt: int -> 30 (rate of creation of pawns)
    creation_rate: list -> len(creation_rate) = 96 (number of people to be created there each 24*3600/96 seconds = 15 min)
    source_location: dict -> {lat: 12.324530, lon: 44.232430}
    pawns_from_weight: dict -> {tourist: {'beta_bp_miss':0.5,'speed_mps':1}}   
    '''
    def __init__(self,
                name = '',
                creation_dt = 30,
                creation_rate = [],
                source_location = {},
                pawns_from_weight = {}
                ):
        self.list_attributes = [name, creation_dt, creation_rate,source_location, pawns_from_weight]
        self.name = name
        self.creation_dt = creation_dt
        self.creation_rate = creation_rate
        self.source_location = source_location
        self.pawns_from_weight = pawns_from_weight
        self.is_added = False
        self.is_default = False
        self.is_reset = False
        self.is_changed = False
        #for locals
        self.source_type = 'ctrl'
        self.pawns = {}
    
    
    
    
class attraction:
    '''Attraction class:
    ----------------
    Input;
    name: str -> T_Fondaco_dei_Tedeschi_by_DFS
    lat: float
    lon: float
    weight: list -> len(list) = 24
    time_cap: list -> len(list) = 24
    visit_time: float
    '''
    def __init__(self,
                 name = '',
                 lat = 0,
                 lon = 0,
                 weight = [],
                 time_cap = [],
                 visit_time = 0
                 
                ):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.weight = weight
        self.time_cap = time_cap
        self.visit_time = visit_time
        self.is_added = False
        self.is_default = False
        self.is_reset = False
        self.is_changed = False
