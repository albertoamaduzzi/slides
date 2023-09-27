#! /usr/bin/env python3
import sys
import os
import json
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')  
try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e


class simulator:
    '''Simulator class:
    -------------
    Input:
    config: str -> /path/to/conf_venezia.json
    config0: str -> /path/to/conf.json.local.albi.make
    start_date: str -> 2021-07-14 23:00:00
    stop_date: str -> 2021-07-21 23:00:00
    average_fluxes: bool -> True
    attraction_activate: bool -> True
    locals: bool -> True -> LOCALS = [0,..,0]
    local_distribution: object -> np.random.normal() -> default none
    new_source_list: str -> ['Scalzi_2','Scalzi_3'] (list of new sources further COSTITUZIONE,PAPPADOPOLI,SALVATORE)
    new_source_bool: bool -> True
    reset_source_list: str -> 'Costituzione_IN-Papadopoli_IN-Schiavoni_IN'
    reset_source_bool: bool -> True
    change_source_list: str -> 'Papadopoli_IN'
    change_source_bool: bool -> True
    name_attractions: str -> 'Farsetti_1_IN' (it must already contain the format present in barrier file)
    add_attractions_bool: bool -> True
    '''
    def __init__(self,enter_parameters_simulation):
# FILE OF INTEREST
        config = enter_parameters_simulation['config']
        config0 = enter_parameters_simulation['config0']
        file_distances_real_data = enter_parameters_simulation['file_distances_real_data']
        dir_data = enter_parameters_simulation['dir_data']
        state_basename = enter_parameters_simulation['state_basename']
        start_date = enter_parameters_simulation['start_date']
        stop_date = enter_parameters_simulation['stop_date']
        average_fluxes = enter_parameters_simulation['average_fluxes']
        attraction_activate = enter_parameters_simulation['attraction_activate']
        locals_ = enter_parameters_simulation['locals_']
        local_distribution = enter_parameters_simulation['local_distribution']
        list_new_source = enter_parameters_simulation['list_new_source']
        list_reset_source = enter_parameters_simulation['list_reset_source']
        list_change_source = enter_parameters_simulation['list_change_source']
        list_new_attractions = enter_parameters_simulation['list_new_attractions']
        list_reset_attractions = enter_parameters_simulation['list_reset_attractions']
        list_change_attractions = enter_parameters_simulation['list_change_attractions']
        cnt_file = enter_parameters_simulation['file_cnt']
        with open(config, encoding='utf8') as f:
            self.simcfgorig = json.load(f)
        with open(config0, encoding='utf8') as g:
            self.simcfgorig0 = json.load(g)
        self.simcfgorig0['work_dir'] = state_basename
        self.df=pd.read_excel(file_distances_real_data, engine='openpyxl')
        self.dir_data = dir_data
        self.data_barriers = pd.read_csv(self.dir_data,';')
    # VALIDATION SET 
        self.average_fluxes = average_fluxes
        self.pick_day = not average_fluxes
        self.attraction_activate = attraction_activate
                    ### SIMULATION SETTINGS ####
    # SAVING DICT
        self.dir_plot = os.path.join(state_basename,'plots')
        self.state_basename = os.path.join(state_basename)
        self.sim_path = ''
        self.path_output_sim = ''
    # DATETIME SETTING
        self.time_format='%Y-%m-%d %H:%M:%S'
        self.start_date = start_date
        self.stop_date = stop_date
    # LOCALS 
        self.locals = locals_
        self.local_distribution = local_distribution
        self.number_population = self.simcfgorig0['model_data']['params']['venezia']['population']
        self.number_daily_tourist = self.simcfgorig0['model_data']['params']['venezia']['daily_tourist']
        self.number_people = self.number_population + self.number_daily_tourist
    # SOURCES  
        self.list_new_sources = list_new_source
        print('simulator initialization:\t',list_new_source,'enter parameter\t',enter_parameters_simulation['list_new_source'])
        self.list_reset_sources = list_reset_source
        self.list_change_sources = list_change_source
    # ATTRACTIONS
        self.list_new_attractions = list_new_attractions
        self.list_reset_attractions = list_reset_attractions
        self.list_change_attractions = list_change_attractions
    # PARALLEL SIMULATION PARAMETERS
        self.create_list_delta_u_attraction()
        self.nsim = len(self.list_delta_u_attraction)
        self.nagents = 6
        self.chunksize = self.nsim // self.nagents
    # PLACE
        self.citytag = 'venezia'    # SETTING THE FOLDERS REQUIRED
    # PC CONFIG
        self.windows = True
    # DUMPA IL FILE DI CONFIGURAZIONE CON LE VARIABILI LOCALI
        self.simcfgorig['file_pro'] = os.path.join(os.environ['WORKSPACE'],'slides','vars','cart','venezia.pro')
        self.simcfgorig['file_pnt'] = os.path.join(os.environ['WORKSPACE'],'slides','vars','cart','venezia.pnt')
        self.simcfgorig['state_basename'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides')
        self.simcfgorig['file_barrier'] = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','barriers_config.csv')
        self.simcfgorig['start_date'] = start_date
        self.simcfgorig['stop_date'] = stop_date
        self.simcfgorig['file_cnt'] = cnt_file
        with open(config, 'w',encoding='utf8') as f:
            json.dump(self.simcfgorig,f,indent = 4)


            ##### HANDLING DF (EXCEL FILE FROM REGION) #######



    def pick_day_input(self):
        '''Description:
        Transforms the excel file in a dataframe columns = (timestamp,NAME_BARRIER_IN, NAME_BARRIER_OUT)
        Where timestamp takes values on the 24 hours after self.start_date
        Is complementary to average_fluxes.
        NOTE: self.df_day is for simulation purposes: it will be used for analyses in the future
        To plot data I need df_day_hrs_inv'''
        if self.pick_day:
            group = self.df.groupby('varco')
            self.df_day = pd.DataFrame()
            df_temp = pd.DataFrame()
#            self.df_day_hrs_inv = pd.DataFrame()
#            df_temp_hrs_inv = pd.DataFrame() 
            for g in group.groups:
                try:
                    # DATAFRAME TO GIVE TO SOURCES FOR SIMULATION
#                    l_= np.array(group.get_group(g)['timestamp'].copy(),dtype='datetime64[s]') #df_temp['timestamp']
#                    df_temp['timestamp'] = l_-np.timedelta64(1, "h")
#                    df_temp[group.get_group(g).iloc[0]['varco']+'_IN'] = np.array(group.get_group(g)['direzione'].copy(),dtype=int)
#                    df_temp[group.get_group(g).iloc[0]['varco']+'_OUT'] = np.array(group.get_group(g)['Unnamed: 3'].copy(),dtype=int)
                    # DATAFRAME FOR ANALYSIS REAL  
                    df_temp['timestamp'] = np.array(group.get_group(g)['timestamp'].copy(),dtype='datetime64[s]') #df_temp_hrs_inv
                    df_temp[group.get_group(g).iloc[0]['varco']+'_IN'] = np.array(group.get_group(g)['direzione'].copy(),dtype=int) #df_temp_hrs_inv
                    df_temp[group.get_group(g).iloc[0]['varco']+'_OUT'] = np.array(group.get_group(g)['Unnamed: 3'].copy(),dtype=int) #df_temp_hrs_inv  
                except:
                    print('shape of df_temp {0} for barrier {1}'.format(np.shape(df_temp),g))
                    pass
                start_date = datetime.strptime(self.start_date,self.time_format)
                stop_date = datetime.strptime(self.stop_date,self.time_format)
                mask_day = [True if h>=start_date and h<=stop_date else False for h in df_temp.timestamp]
                try:
                    # DATAFRAME FOR ANALYSIS OF REAL DATA
                    self.df_day['timestamp'] = df_temp.loc[mask_day]['timestamp']    #df_day_hrs_inv
                    self.df_day[group.get_group(g).iloc[0]['varco']+'_IN'] = df_temp[mask_day][group.get_group(g).iloc[0]['varco']+'_IN'] #df_day_hrs_inv
                    self.df_day[group.get_group(g).iloc[0]['varco']+'_OUT'] = df_temp[mask_day][group.get_group(g).iloc[0]['varco']+'_OUT'] #df_day_hrs_inv
                    # DATAFRAME TO GIVE TO THE SIMULATION
#                    self.df_day['timestamp'] = df_temp.loc[mask_day]['timestamp']  
#                    self.df_day[group.get_group(g).iloc[0]['varco']+'_IN'] = df_temp[mask_day][group.get_group(g).iloc[0]['varco']+'_IN']
#                    self.df_day[group.get_group(g).iloc[0]['varco']+'_OUT'] = df_temp[mask_day][group.get_group(g).iloc[0]['varco']+'_OUT']
                except KeyError:
                    print(g,'in not found as a barrier')
                self.tags = list(df_temp.columns.drop('timestamp'))
                self.tagn = len(self.tags)
            if self.windows:
#                self.df_day_hrs_inv.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}_hrs_inv.csv'.format(self.start_date.split(' ')[0]),';')                
                self.df_day.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(self.start_date.split(' ')[0]),';')
            else:
#                self.df_day_hrs_inv.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'\\dataframe_real_data_pick_day_{}_hrs_inv.csv'.format(self.start_date.split(' ')[0]),';')                
                self.df_day.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'\\dataframe_real_data_pick_day_{}.csv'.format(self.start_date.split(' ')[0]),';')
        else:
            print('This method is useless as I have chosen to average fluxes')

        return self

    def averaging_fluxes(self):
        '''Description:
        Transforms the excel file in a dataframe columns = (timestamp,NAME_BARRIER_IN, NAME_BARRIER_OUT)
        Where timestamp takes values on the 24 hours after self.start_date
        averaging them on the week or weekend day'''
        if self.average_fluxes:
            group=self.df.groupby('varco')
            df_temp = pd.DataFrame()
 #           df_temp_hrs_inv = pd.DataFrame() 

            for g in group.groups:
                try:
                    # DATAFRAME DA DARE IN PASTO ALLA SIMULAZIONE IN CUI I DATI SONO ANTICIPATI DI 1 ORA PER CREARLI E MISURARLI ALL'ORARIO ESATTO
#                    l_= np.array(group.get_group(g)['timestamp'].copy(),dtype='datetime64[s]') #df_temp['timestamp']
#                    df_temp['timestamp'] = l_-np.timedelta64(1, "h")
#                    df_temp[group.get_group(g).iloc[0]['varco']+'_IN'] = np.array(group.get_group(g)['direzione'].copy(),dtype=int)
#                    df_temp[group.get_group(g).iloc[0]['varco']+'_OUT'] = np.array(group.get_group(g)['Unnamed: 3'].copy(),dtype=int)
                    # DATAFRAME DA TENERLI PER LE ANALISI
                    df_temp['timestamp']=np.array(group.get_group(g)['timestamp'].copy(),dtype='datetime64[s]') #df_temp_hrs_inv
                    df_temp[group.get_group(g).iloc[0]['varco']+'_IN'] = np.array(group.get_group(g)['direzione'].copy(),dtype=int) #df_temp_hrs_inv
                    df_temp[group.get_group(g).iloc[0]['varco']+'_OUT'] = np.array(group.get_group(g)['Unnamed: 3'].copy(),dtype=int) #df_temp_hrs_inv  
                except:
                    pass
            weekend_mask = [True if h.weekday()>4 else False for h in df_temp.timestamp]
            week_mask = [True if h.weekday()<4 else False for h in df_temp.timestamp]
            if len(weekend_mask) == 0:
                print('We are averaging during over week days')
                df_temp = df_temp.loc[week_mask]
#                df_temp_hrs_inv = df_temp_hrs_inv.loc[week_mask]
                h_mask = np.arange(24)
                self.tags = list(df_temp.columns.drop('timestamp'))
                self.tagn = len(self.tags)
                self.df_avg = pd.DataFrame(data=tuple(np.empty([24,self.tagn])),columns = self.tags)
#                self.df_avg_hrs_inv = pd.DataFrame(data=tuple(np.empty([24,self.tagn])),columns = self.tags)
                count_hour = 0
                for e in h_mask:
                    mask = [True if h.hour==e else False for h in df_temp.timestamp]                                          
                    self.df_avg.iloc[e][self.tags] = df_temp[self.tags].loc[mask].mean()
 #                   self.df_avg_hrs_inv.iloc[e][self.tags] = df_temp_hrs_inv[self.tags].loc[mask].mean()
                time_ = np.unique([h.hour if h.hour else False for h in df_temp.timestamp]).tolist()
                time_hrs_inv = np.unique([h.hour if h.hour else False for h in df_temp_hrs_inv.timestamp]).tolist()
                
                self.df_avg['timestamp'] = time_
#                self.df_avg_hrs_inv['timestamp'] = time_hrs_inv
                
                if self.windows:
                    self.df_avg.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_avg_week_days{}.csv'.format(self.start_date.split(' ')[0]),';')
#                    self.df_avg_hrs_inv.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_avg_week_days{}_hrs_inv.csv'.format(self.start_date.split(' ')[0]),';')
                else:
                    self.df_avg.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'\\dataframe_real_data_avg_week_days{}.csv'.format(self.start_date.split(' ')[0]),';')                
#                    self.df_avg_hrs_inv.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'\\dataframe_real_data_avg_week_days{}_hrs_inv.csv'.format(self.start_date.split(' ')[0]),';')                
            else:
                print('We are averaging over week-end days')
                df_temp = df_temp.loc[weekend_mask]
                df_temp_hrs_inv = df_temp_hrs_inv.loc[weekend_mask]
                h_mask = np.arange(24)
                self.tags = list(df_temp.columns.drop('timestamp'))
                self.tagn = len(self.tags)
                self.df_avg = pd.DataFrame(data=tuple(np.empty([24,self.tagn])),columns = self.tags)
                self.df_avg_hrs_inv = pd.DataFrame(data=tuple(np.empty([24,self.tagn])),columns = self.tags)
                for e in h_mask:
                    mask = [True if h.hour==e else False for h in df_temp.timestamp]                                          
                    self.df_avg.iloc[e][self.tags] = df_temp[self.tags].loc[mask].mean()
                    self.df_avg_hrs_inv.iloc[e][self.tags] = df_temp_hrs_inv[self.tags].loc[mask].mean()

                time_ = np.unique([h.hour if h.hour else False for h in df_temp.timestamp]).tolist()
                time_hrs_inv = np.unique([h.hour if h.hour else False for h in df_temp_hrs_inv.timestamp]).tolist()

                self.df_avg['timestamp'] = time_   
                self.df_avg_hrs_inv['timestamp'] = time_hrs_inv
 
                if self.windows:
                    self.df_avg.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_avg_weekend_days{}.csv'.format(self.start_date.split(' ')[0]),';')
                    self.df_avg_hrs_inv.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_avg_weekend_days{}_hrs_inv.csv'.format(self.start_date.split(' ')[0]),';')

                else:
                    self.df_avg.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'\\dataframe_real_data_avg_weekend_days{}.csv'.format(self.start_date.split(' ')[0]),';')
                    self.df_avg_hrs_inv.to_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'\\dataframe_real_data_avg_weekemd_days{}_hrs_inv.csv'.format(self.start_date.split(' ')[0]),';')                

        return self 

    def normalize_fluxes(self):
        '''Normalizes fluxes and creates df_norm_fluxes
        That has columns (timestamp,BARRIER_IN,BARRIER_OUT)
        timestamp contains just the hours if averaged'''
        if self.average_fluxes:
            self.df_norm_fluxes = self.df_avg.drop(columns = 'timestamp').apply(lambda x: x/sum(x))
        else:
            self.df_norm_fluxes = self.df_day.drop(columns = 'timestamp').apply(lambda x: x/sum(x))


    def initialize_standard_correlation_matrix(self):
        '''Input:
        ------------------------
        NOTE: there is the need of a bigger dataset rather then 1 day if we want to study the dynamic of the largest eigenvalue
        That is the one proportional to the average of all elements in the matrix.
        Object simulation:
        Is the matrix from Schreckenberg paper for which it is possible to capture the dominant effect
        by the largest eigenvalue. The proble is to define the right T (here 24 for our dataset) as
        if T -> too small, then, too many fluctuations
        if T -> too large '''
        if self.average_fluxes:
            self.std_correlation_matrix_real_data = self.df_avg.drop(columns = 'timestamp').apply(lambda x: (x - x.mean())/np.std(x)) 
        else:
            self.std_correlation_matrix_real_data = self.df_day.drop(columns = 'timestamp').apply(lambda x: (x - x.mean())/np.std(x))

    def initialize_A(self):
        '''Input:
        ------------------------
        Object simulation:
        Is the matrix from Schreckenberg paper for which I apply the singular value decomposition'''
        if self.average_fluxes:
            self.A_real_data = self.df_avg.drop(columns = 'timestamp').apply(lambda x: x - x.mean()) 
        else:
            self.A_real_data = self.df_day.drop(columns = 'timestamp').apply(lambda x: x - x.mean())



    def create_list_delta_u_attraction(self):
        '''Create the list of modified attractions:
        Need to find a criterion to give to choose how to give these'''

        attraction_perturbation_activate= self.attraction_activate
        if attraction_perturbation_activate: 
            self.delta_u = list(np.linspace(-0.3,0.3,10))
            self.attractions = list(self.simcfgorig['attractions'].keys())# I am returned with the list of attractions
        else:
            self.delta_u = [0]
            self.attractions = list(self.simcfgorig['attractions'].keys())# I am returned with the list of attractions
  # modifico i parametri di interesse (attractivity) 
        self.list_delta_u_attraction=[]
        for attraction in self.attractions:
            for w in self.delta_u:
                self.list_delta_u_attraction.append([attraction,w])
        return self
    

    def assign_directory_state_basename(self,dict_sources):
        '''Input:
        ------------------
        directory_sources: str -> from the configuration handler class defines the ../work_slides as basename
        This is the position where the new simulation is set. NOTE>
        self.state_basename is where I put the results of the successive analysis.
        However I should have also'''
        sourcs = 'src_'
        attrs = 'attr_'
        count_src = 0
        count_attrs = 0
        for k in list(dict_sources.keys()):
            if dict_sources[k].is_reset:
                pass
            else:
                sourcs = sourcs + k + '-'
        for k in list(self.list_new_attractions):
            attrs = attrs + k + '-'
        self.sim_path = sourcs +'--' + attrs
        if self.pick_day:
            self.sim_path = self.sim_path + self.start_date.split(' ')[0] + '_' + self.start_date.split(' ')[1].split(':')[0] 
        self.state_basename = os.path.join(self.state_basename,self.sim_path)
        if not os.path.exists(self.state_basename):
            os.mkdir(self.state_basename)

    def run_sim(self):
        try:
            sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
            from pysim import simulation

            sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
            from conf import conf
        except Exception as e:
            raise Exception('library loading error : {}'.format(e)) from e

        # lancio la simulazione
        cfg = conf(self.simcfgorig0)
        simcfg = cfg.generate(start_date = self.start_date, stop_date = self.stop_date, citytag = self.citytag)#(override_weights=w)
        # Adding piece where I change the sources instead of conf.py
        simcfgs = json.dumps(self.simcfgorig)
        s = simulation(simcfgs)
        print(s.sim_info())
        s.run()
        return True  



#    def dictionary_per_type(self):
#        '''USAGE:
#        Dictionaries to initialize non-uniform weights 
#        with different distribution '''
#        self.dict_distributions = {'uniform': np.random.uniform(24).tolist()}
#        self.dict_attractions = {'name': self.list_new_attractions,}
#        self.df_new_attracrions = 
