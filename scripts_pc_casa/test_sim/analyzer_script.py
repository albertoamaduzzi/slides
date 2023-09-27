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
import scipy.spatial.distance as ssd
from scipy.signal import savgol_filter
import scipy.cluster.hierarchy as sch

warnings.filterwarnings('ignore')  

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

class analyzer:
    '''Analyzer class:
    ----------------------
    Input;
    start_date: str -> 210718 (is start_date di simulation -> to build the name of data_barriers\venezia_barriers_()_000000)
    sim_fluxes_file: str -> sim.state_basename + \attr_...\sources_\no_locals ()
    dir_comparison: str -> 
    df_barriers: dataframe -> containing csv with daily fluxess produced by simulation
    NOTE:
    change windows to change the way directories are written
    '''
    def __init__(self,
                 sim_,n_epoch):
        self.windows = True
        self.dir_plot = sim_.dir_plot
        self.sim_date = sim_.start_date
        print('analyzer: start date simulation',sim_.start_date)
#        self.df_barriers = sim_.data_barriers
        self.str0 = sim_.start_date.split('-')[0][-2:] + sim_.start_date.split('-')[1] + sim_.start_date.split('-')[2].split(' ')[0] 
        self.str1 = sim_.start_date.split('-')[2].split(' ')[1].split(':')[0] +sim_.start_date.split('-')[2].split(' ')[1].split(':')[1] + sim_.start_date.split('-')[2].split(' ')[1].split(':')[2]
        print('analyzer: dates to read the simulated csv',self.str0,self.str1)
        if self.windows:
            self.sim_dataframe = pd.read_csv(sim_.path_output_sim +'/venezia_barriers_{0}_{1}.csv'.format(self.str0,self.str1),';') 
        else:
            self.sim_dataframe = pd.read_csv(sim_.path_output_sim +'\\venezia_barriers_{0}_{1}.csv'.format(self.str0,self.str1),';') 
            
        if sim_.pick_day:
            self.df_barriers = sim_.df_day            
        else:
            self.df_barriers = sim_.df_avg
        self.n_epoch = n_epoch
            
        
    def create_dict_barrier_sim_real(self,sim):
        '''Input:
        ---------------------------
        sim: simulation object
        DESCRIPTION VARIABLES:
        -------------------------
        self.dict_barrier_real: dict -> dict_barrier_real[name]: dict -> dict_barrier_real[name][hour]
        self.dict_barrier_sim: dict -> dict_barrier_sim[name]: dict -> dict_barrier_sim[name][hour]
        name <- self.sim_dataframe.columns, self.df_barriers.columns
        hour <- self.df_barriers['timestamp'], self.sim_dataframe['datetime']
        '''
        self.dict_barrier_sim = dict.fromkeys(list(self.sim_dataframe.columns.drop(['datetime','timestamp']))) 
        self.dict_barrier_real = dict.fromkeys(list(self.sim_dataframe.columns.drop(['datetime','timestamp'])))    
        for b_ in self.sim_dataframe.columns.drop(['datetime','timestamp']):
            b = barrier()
            b.name = b_
            b.flux = self.sim_dataframe[['datetime',b.name]].set_index('datetime').to_dict()
            self.dict_barrier_sim[b_] = b
        for b_ in self.df_barriers.columns.drop('timestamp'):
            b = barrier()
            b.name = b_
            b.flux = self.df_barriers[['timestamp',b.name]].set_index('timestamp').to_dict()
            self.dict_barrier_real[b_] = b
            
              
            
            
    def produce_comparison_df(self,sim_):
        '''Input:
        -----------------
        sim_: simulation object -> sim_.df_avg if sim_pick_day = False, sim_.df_day if True
        self.path_comparison: directory where to save the comparison  
        Output:
        self.df_comparison_sim_data: dataframe columns ['barrier','time','in_sim','out_sim','in+out_sim','in_data','out_data','in+out_data'] NOT USED
        '''
        self.create_dict_barrier_sim_real(sim_)
        # output
        list_in_data = []
        list_out_data = []
        list_in_out_data = []
        list_in_sim = []
        list_out_sim = []
        list_in_out_sim = []
        list_time = []
        list_barriere = []

        self.dictionary_comparison = {'barrier':[],'time':[],'in_sim':[],'out_sim':[],'in+out_sim':[],'in_data':[],'out_data':[],'in+out_data':[]}
        self.df_comparison_sim_data = pd.DataFrame(self.dictionary_comparison)
        list_name_barriers = []
        for k in list(self.dict_barrier_real.keys()):
            a = k.split('_')[0] + '_' + k.split('_')[1]
            list_name_barriers.append(a)
        for nb in list_name_barriers:
            dsim_in = self.dict_barrier_sim[nb + '_IN']
            dreal_in = self.dict_barrier_real[nb + '_IN']
            dsim_out = self.dict_barrier_sim[nb + '_OUT']
            dreal_out = self.dict_barrier_real[nb + '_OUT']
            for h in range(len(dreal_out)):
                list_in_data.append(int(dreal_in[h]))
                list_in_sim.append(int(dsim_in[h]))
                list_time.append(h)
                list_out_data.append(int(dreal_out[h]))
                list_out_sim.append(int(dsim_out[h]))
                list_in_out_sim.append(int(dsim_out[h]) + int(dsim_in[h]))
                list_in_out_data.append(int(dreal_out[h]) + int(dreal_in[h]))
                list_barriere.append(nb)
        self.df_comparison_sim_data['barrier']=list_barriere
        self.df_comparison_sim_data['time']=pd.Series(list_time)
        self.df_comparison_sim_data['in_sim']=list_in_sim
        self.df_comparison_sim_data['out_sim']=list_out_sim
        self.df_comparison_sim_data['in+out_sim']=list_in_out_sim
        self.df_comparison_sim_data['in_data']=list_in_data
        self.df_comparison_sim_data['out_data']=list_out_data
        self.df_comparison_sim_data['in+out_data']=list_in_out_data
        self.df_comparison_sim_data.to_csv(os.path.join(sim_.state_basename,'df_comparison_sim_data_{0}_{1}.csv'.format(sim_.number_people,self.str0)),';')
        self.path_comparison = os.path.join(sim_.state_basename,'df_comparison_sim_data_{0}_{1}.csv'.format(sim_.number_people,self.str0))
            
    
    def distance_csv_for_ward(self,sim,dict_dir):
        '''Input:
        ---------------------
        sim: simulation object
        Description:
        ----------------------
        saves df_dist_sim,df_norm_dist_sim, df_dist_real and df_norm_dist_real -> columns ['tag1','tag2','barrier1','barrier2','day','distance']
        They contain the Ward data for 1 day or the averaged period depending on the simulation.
        Here there is an allineation between simulation and real data.
        self.dict_df_ward contains the files [barrier1,barrier2,tag1,tag2,distance] for each simulation and real data,
        both normalized and no
        Output:
        ----------------------
        dict_df_ward: dict -> keys() = 'df_dist_sim','df_dist_sim_norm','df_dist_real','df_dist_real_norm'
                           -> values() =  df_dist_sim, df_dist_sim_norm, df_dist_real, df_dist_real_norm : columns -> ['tag1','tag2','barrier1','barrier2','distance']

        SAVED ON: sim.state_basename + '/normed_flux_euclidean_distance_avg_{}_sim.json'.format(self.str0)
        '''
        def norm(a):
            tot = a.sum()
            if tot!=0: return a/tot
            else:      return a

        def dist(a,b,t='euclidean'):
            if t == 'euclidean':
                return np.sqrt(np.sum((a-b)**2)) # or   ssd.euclidean(a,b)
            elif t == 'correlation':
                return ssd.correlation(a,b)
        self.initialize_standard_correlation_matrix()
        self.initialize_A()
        self.normalize_fluxes()
        t_dist_real = []
        t_dist_real_norm = []
        t_dist_sim = []
        t_dist_sim_norm = []
        tags = sim.tags
        tagn = sim.tagn
        dist_type = 'euclidean'
        for j in range(0,sim.tagn-1):
            for k in range(j+1,sim.tagn):
                try:
                    dist_sim = dist(norm(self.sim_dataframe[tags[j]].values),norm(self.sim_dataframe[tags[k]].values),dist_type)
                    norm_dist_sim = dist(norm(self.df_norm_fluxes_sim[tags[j]].values),norm(self.df_norm_fluxes_sim[tags[k]].values),dist_type)                 
                    if sim.average_fluxes:
                        norm_dist_real = dist(norm(sim.df_norm_fluxes[tags[j]].values),norm(sim.df_norm_fluxes[tags[k]].values),dist_type)                 
                        dist_real = dist(norm(sim.df_avg[tags[j]].values),norm(sim.df_avg[tags[k]].values),dist_type)
                        

                    else:
                        norm_dist_real = dist(norm(sim.df_norm_fluxes[tags[j]].values),norm(sim.df_norm_fluxes[tags[k]].values),dist_type)                 
                        dist_real = dist(norm(sim.df_day[tags[j]].values),norm(sim.df_day[tags[k]].values),dist_type)
                except KeyError: 
                    pass
#                    print(tags[k],'is not in the list of barriers produced by the simulation')
                t_dist_real.append([dist_real,j,k,sim.start_date])
                t_dist_real_norm.append([norm_dist_real,j,k,sim.start_date])
                t_dist_sim.append([dist_sim,j,k,sim.start_date])
                t_dist_sim_norm.append([norm_dist_sim,j,k,sim.start_date])

        ### REAL DATA ####
        t_dist_real_norm = np.array(t_dist_real_norm)
        t_dist_real = np.array(t_dist_real)    
        df_dist_real_norm = pd.DataFrame()
        df_dist_real = pd.DataFrame()
        df_dist_real_norm['tag1']= list(t_dist_real_norm[:,1].astype(int))
        df_dist_real_norm['tag2']= list(t_dist_real_norm[:,2].astype(int))#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_real_norm['barrier1'] = [tags[int(a)] for a in df_dist_real_norm['tag1']]
        df_dist_real_norm['barrier2'] = [tags[int(a)] for a in df_dist_real_norm['tag2']]
        df_dist_real_norm['day'] = list(t_dist_real_norm[:,3])
        df_dist_real_norm['distance'] = list(t_dist_real_norm[:,0])
#        df_dist_real_norm = df_dist_real_norm.sort_values(by='distance')
        df_dist_real['tag1']= list(t_dist_real[:,1].astype(int))
        df_dist_real['tag2']= list(t_dist_real[:,2].astype(int))#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_real['barrier1'] = [tags[int(a)] for a in df_dist_real['tag1']]
        df_dist_real['barrier2'] = [tags[int(a)] for a in df_dist_real['tag2']]
        df_dist_real['day'] = list(t_dist_real[:,3])
        df_dist_real['distance'] = list(t_dist_real[:,0])
#        df_dist_real = df_dist_real.sort_values(by='distance')
#        df_dist_real_norm = df_dist_real_norm.sort_values(by='distance')
        self.df_dist_real = df_dist_real.to_dict()
        self.df_dist_real_norm = df_dist_real_norm.to_dict()
#        if sim.average_fluxes:
#            df_dist_real.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg.csv'),';')
#            df_dist_real_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg.csv'),';')
#        else:
#            df_dist_real.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg_{}.csv'.format(self.str0)),';')
#            df_dist_real_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg_{}.csv'.format(self.str0)),';')
        #### SIMULATION ####
        df_dist_sim_norm = pd.DataFrame()
        df_dist_sim = pd.DataFrame()
        t_dist_sim_norm = np.array(t_dist_sim_norm)
        t_dist_sim = np.array(t_dist_sim)
        df_dist_sim_norm['tag1']= list(t_dist_sim_norm[:,1].astype(int))
        df_dist_sim_norm['tag2']= list(t_dist_sim_norm[:,2].astype(int))#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_sim_norm['barrier1'] = [tags[int(a)] for a in df_dist_real_norm['tag1']]
        df_dist_sim_norm['barrier2'] = [tags[int(a)] for a in df_dist_real_norm['tag2']]
        df_dist_sim_norm['day'] = list(t_dist_sim_norm[:,3])
        df_dist_sim_norm['distance'] = list(t_dist_sim_norm[:,0])
#        df_dist_sim_norm = df_dist_sim_norm.sort_values(by='distance')
        df_dist_sim['tag1']= list(t_dist_sim[:,1].astype(int))
        df_dist_sim['tag2']= list(t_dist_sim[:,2].astype(int))#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_sim['barrier1'] = [tags[int(a)] for a in df_dist_real['tag1']]
        df_dist_sim['barrier2'] = [tags[int(a)] for a in df_dist_real['tag2']]
        df_dist_sim['day'] = list(t_dist_sim[:,3])
        df_dist_sim['distance'] = list(t_dist_sim[:,0])
#        df_dist_sim = df_dist_sim.sort_values(by='distance')
#        df_dist_sim_norm = df_dist_sim_norm.sort_values(by='distance')
        self.df_dist_sim = df_dist_sim.to_dict()
        self.df_dist_sim_norm = df_dist_sim_norm.to_dict()
#        if sim.average_fluxes:
#            df_dist_sim.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg_sim.csv'),';')
#            df_dist_sim_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg_sim.csv'),';')
#        else:
#            df_dist_sim.to_csv(os.path.join(sim.state_basename,'euclidean_distance_{}_sim.csv'.format(self.str0)),';')
#            df_dist_real_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_{}_sim.csv'.format(self.str0)),';')
        dict_df_ward = {'df_dist_sim':self.df_dist_sim,'df_dist_sim_norm':self.df_dist_sim_norm,'df_dist_real':self.df_dist_real,'df_dist_real_norm':self.df_dist_real_norm}
        self.dict_df_ward = {'df_dist_sim':df_dist_sim,'df_dist_sim_norm':df_dist_sim_norm,'df_dist_real':df_dist_real,'df_dist_real_norm':df_dist_real_norm}
        if sim.average_fluxes:
            if self. windows:
                with open(sim.state_basename + dict_dir + '/dict_euclidean_distances_avg_{}_sim.json'.format(self.str0),'w') as outfile:
                    json.dump(dict_df_ward,outfile)
            else:
                with open(sim.state_basename + dict_dir + '\\dict_euclidean_distances_avg_{}_sim.json'.format(self.str0),'w') as outfile:
                    json.dump(dict_df_ward,outfile)

        else:
            if self. windows:
                with open(sim.state_basename + dict_dir + '/dict_euclidean_distances_{}_sim.json'.format(self.str0),'w') as outfile:
                    json.dump(dict_df_ward,outfile)                
            else:
                with open(sim.state_basename + dict_dir + '\\dict_euclidean_distances_{}_sim.json'.format(self.str0),'w') as outfile:
                    json.dump(dict_df_ward,outfile)                

    def pearson_corr(self,x,y):
        xxu = np.array(x) - np.array(x).mean()
        yyu = np.array(y)-np.array(y).mean()
        return np.dot(xxu,yyu)/len(np.array(x))/(np.std(np.array(x)*np.std(np.array(y))))

        

    def correlation_matrix_plot(self,sim,plots_):
        '''Input:
        sim: sim_dataframe, df_barriers
        plot_: directory /plot_n where n would be an index for an iteration.
        ---------------------
        Description:
        1) Calculate the correlation matrices between different simulations (corrMatrix,corrMatrix1)
        2) Compare barriers real-sim
        2.1) dict_corr_same_barrier (_scaled): keys = barrier (i.e. APOSTOLI_1_IN), values pearson_corr_{t} (n_it(real),n_it(sim))
        2.2) compare these values with <p_corr>_{i} if p_corr_i > <p_corr>_{i} -> we have top reproduced barrier
        2.3)  
        Output:
        ----------------------
        dif_cor: correlation matrix for plots in the classification and plotting section, correlation_distance sim real attributes
        cost_corr: is the one that every iteration changes and helps me find the minimum
        Description:
        Produce the correlation matrices among different barriers for both simulations and real data in both normalized and non normalized case.
        Produce correlation matrix among same barriers of real and simulated case.
                self.list_good_corr = []
        self.list_good_corr_good_sim = []
        self.list_good_corr_bad_sim = []        
        self.list_bad_corr = []
        self.list_bad_corr_bad_sim = []
        self.list_bad_corr_good_sim = []
        are good for discriminating among barriers that correlate well (for just toursim)
'''
        cols_to_drop_sim = set(self.sim_dataframe.columns).difference(set(self.df_barriers.columns))
        cols_to_drop_real = set(self.df_barriers.columns).difference(set(self.sim_dataframe.columns))
#        print('cols_to_drop_sim',list(cols_to_drop_sim))
#        print('cols_to_drop_real',list(cols_to_drop_real))
        if len(list(cols_to_drop_sim))!=0:
            list(cols_to_drop_sim).append('datetime')
            list(cols_to_drop_sim).append('timestamp')
            cols_to_drop_sim = np.unique(list(cols_to_drop_sim))            
            corrMatrix = self.sim_dataframe.drop(cols_to_drop_sim,axis = 1).corr()
            corrMatrix_norm = self.df_norm_fluxes_sim.drop(cols_to_drop_sim,axis = 1).corr()
            # Correlation real matrix
            self.tmp_flux_sim = self.sim_dataframe.drop(cols_to_drop_sim,axis = 1)
            self.tmp_flux_sim_norm = self.df_norm_fluxes_sim.drop(cols_to_drop_sim,axis = 1)

        else:
            corrMatrix = self.sim_dataframe.drop(['datetime','timestamp'],axis = 1).corr()
            corrMatrix_norm = self.df_norm_fluxes_sim.drop(['datetime','timestamp'],axis = 1).corr()
            # Correlation real sim
            self.tmp_flux_sim = self.sim_dataframe.drop(cols_to_drop_sim,axis = 1).corr()
            self.tmp_flux_sim_norm = self.df_norm_fluxes_sim.drop(cols_to_drop_sim,axis = 1).corr()

        if len(list(cols_to_drop_real))!=0:
            list(cols_to_drop_real).append('timestamp')
            cols_to_drop_real = np.unique(list(cols_to_drop_real))
            corrMatrix1 = self.df_barriers.drop(cols_to_drop_real,axis = 1).corr()
            corrMatrix1_norm = sim.df_norm_fluxes.drop(cols_to_drop_real,axis = 1).corr()
            # Correlation real sim
            self.tmp_flux_real = self.df_barriers.drop(cols_to_drop_real,axis = 1)
            self.tmp_flux_real_norm = sim.df_norm_fluxes.drop(cols_to_drop_real,axis = 1)
        else:
            corrMatrix1 = self.df_barriers.drop('timestamp',axis = 1).corr()
            corrMatrix1_norm = sim.df_norm_fluxes.drop('timestamp',axis = 1).corr()
            # For Correlation real sim 
            self.tmp_flux_real = self.df_barriers.drop(cols_to_drop_real,axis = 1)
            self.tmp_flux_real_norm = sim.df_norm_fluxes.drop(cols_to_drop_real,axis = 1)
        if self.windows:
            if self.n_epoch == 0:
                plots_ = '/plots_{}/'.format(0)
            else:
                plots_ = plots_ + '/plots_{}/'.format(self.n_epoch)
        else:
            if self.n_epoch == 0:
                plots_ = '\\plots_{}\\'.format(0)
            else:
                plots_ = plots_ + '\\plots_{}\\'.format(self.n_epoch)
        if not os.path.exists(sim.state_basename + plots_):
            os.mkdir(sim.state_basename + plots_)        
#### CORRELATION SAME BARRIER SIM-REAL
        dict_corr_same_barrier = dict.fromkeys(self.tmp_flux_real.drop('timestamp',axis = 1).columns)
        dict_corr_same_barrier_norm = dict.fromkeys(self.tmp_flux_real_norm)
#### CORRELATION SAME BARRIER SIM-REAL SCALED
        dict_corr_same_barrier_scaled = dict.fromkeys(self.tmp_flux_real.drop('timestamp',axis = 1).columns)

#        print(self.tmp_flux_real_norm.keys(),self.tmp_flux_real.keys())
        for k in dict_corr_same_barrier:
            dict_corr_same_barrier[k] = self.pearson_corr(self.tmp_flux_real.drop('timestamp',axis = 1)[k],self.tmp_flux_sim.drop('timestamp',axis = 1)[k])
            dict_corr_same_barrier_norm[k] = self.pearson_corr(self.tmp_flux_real_norm[k],self.tmp_flux_sim_norm[k])
            dict_corr_same_barrier_scaled[k] = self.pearson_corr(self.tmp_flux_real[k]/sum(self.tmp_flux_real[k]),self.tmp_flux_sim[k]/sum(self.tmp_flux_sim[k]))
                     
# CORRELATION SIM-REAL PER BARRIER
        avg = np.array(list(dict_corr_same_barrier.values()))[~np.isnan(np.array(list(dict_corr_same_barrier.values())))].mean()
        avg_scaled = np.array(list(dict_corr_same_barrier_scaled.values()))[~np.isnan(np.array(list(dict_corr_same_barrier_scaled.values())))].mean()
        std_ = np.std(np.array(list(dict_corr_same_barrier.values()))[~np.isnan(np.array(list(dict_corr_same_barrier.values())))])
        std_scaled = np.std(np.array(list(dict_corr_same_barrier_scaled.values()))[~np.isnan(np.array(list(dict_corr_same_barrier_scaled.values())))])
        
        sx = avg -std_        
        dx = avg + std_
        sx_scaled = avg_scaled -std_scaled        
        dx_scaled = avg_scaled + std_scaled
# PLOT CORRELATION BAR PLOT         
        h = list(np.arange(22))
        fig = plt.figure(figsize=(30, 15))
        plt.bar(x = list(dict_corr_same_barrier.keys()),height = list(dict_corr_same_barrier.values()))
        plt.axhline(avg, color = 'r', linewidth=1)
        plt.xticks(rotation = 90)
        plt.title('Correlation same barrier sim-real data')
        if self.windows:
            if not os.path.exists(sim.state_basename + plots_ + '/correlation_distribution/'):
                os.mkdir(sim.state_basename + plots_ + '/correlation_distribution/')
            self.dir_corr = sim.state_basename + plots_ + '/correlation_distribution/'
                
        else:
            if not os.path.exists(sim.state_basename + plots_ + '\\correlation_distribution\\'):
                os.mkdir(sim.state_basename + plots_ + '\\correlation_distribution\\')
            self.dir_corr = sim.state_basename + plots_ + '\\correlation_distribution\\'                
        plt.savefig(self.dir_corr + 'corr_same_barrier.png')
# SCALED CASE        
        fig = plt.figure(figsize=(30, 15))
        plt.bar(x = list(dict_corr_same_barrier_scaled.keys()),height = list(dict_corr_same_barrier_scaled.values()))
        plt.axhline(avg, color = 'r', linewidth=1)
        plt.xticks(rotation = 90)
        plt.title('Correlation same barrier sim-real data scaled')
        if self.windows:
            if not os.path.exists(sim.state_basename + plots_ + '/correlation_distribution_scaled/'):
                os.mkdir(sim.state_basename + plots_ + '/correlation_distribution_scaled/')
            self.dir_corr_scaled = sim.state_basename + plots_ + '/correlation_distribution_scaled/'
                
        else:
            if not os.path.exists(sim.state_basename + plots_ + '\\correlation_distribution_scaled\\'):
                os.mkdir(sim.state_basename + plots_ + '\\correlation_distribution_scaled\\')
            self.dir_corr_scaled = sim.state_basename + plots_ + '\\correlation_distribution_scaled\\'                
        plt.savefig(self.dir_corr_scaled + 'corr_same_barrier_scaled.png')

# HISTO        
        fig = plt.figure(figsize=(30, 15))
        result = plt.hist(list(dict_corr_same_barrier.values()),50,density = True ,facecolor='g', alpha=0.75)
        plt.axvline(avg, color = 'r',linestyle = '--', linewidth=1)
        plt.axvline(sx,color = 'b',linestyle = '--', linewidth=1)
        plt.axvline(dx,color = 'b',linestyle = '--', linewidth=1)
        plt.xticks(rotation = 90)
        plt.title('Distribution correlation same barrier sim-real data')
        plt.savefig(self.dir_corr + 'distibution_correlation_sim_real.png')
# HISTO SCALED CASE 
        fig = plt.figure(figsize=(30, 15))
        result = plt.hist(list(dict_corr_same_barrier_scaled.values()),50,density = True ,facecolor='g', alpha=0.75)
        plt.axvline(avg_scaled, color = 'r',linestyle = '--', linewidth=1)
        plt.axvline(sx_scaled,color = 'b',linestyle = '--', linewidth=1)
        plt.axvline(dx_scaled,color = 'b',linestyle = '--', linewidth=1)
        plt.xticks(rotation = 90)
        plt.title('Distribution correlation same barrier sim-real data scaled')
        plt.savefig(self.dir_corr_scaled + 'distibution_correlation_sim_real_scaled.png')

# GOOD BAD BARRIERS FOR LOCALS:
        self.list_good_corr = []
        self.list_good_corr_good_sim = []
        self.list_good_corr_bad_sim = []        
        self.list_bad_corr = []
        self.list_bad_corr_bad_sim = []
        self.list_bad_corr_good_sim = []
        self.list_good_corr_scaled = []
        self.list_good_corr_good_sim_scaled = []
        self.list_good_corr_bad_sim_scaled = []        
        self.list_bad_corr_scaled = []
        self.list_bad_corr_bad_sim_scaled = []
        self.list_bad_corr_good_sim_scaled = []
        self.list_qualitative_corr = [self.list_good_corr,self.list_good_corr_good_sim,self.list_good_corr_bad_sim,self.list_bad_corr,self.list_bad_corr_bad_sim,self.list_bad_corr_good_sim]
        self.list_qualitative_corr_scaled = [self.list_good_corr_scaled,self.list_good_corr_good_sim_scaled,self.list_good_corr_bad_sim_scaled,self.list_bad_corr_scaled,self.list_bad_corr_bad_sim_scaled,self.list_bad_corr_good_sim_scaled]
        dict_avg_diff = dict.fromkeys(list(dict_corr_same_barrier.keys())) # bigger then 0 if sim > data
        dict_similarity = dict.fromkeys(list(dict_corr_same_barrier.keys()))
        dict_avg_diff_scaled = dict.fromkeys(list(dict_corr_same_barrier.keys())) # bigger then 0 if sim > data
        dict_similarity_scaled = dict.fromkeys(list(dict_corr_same_barrier.keys()))

        for barr in list(dict_corr_same_barrier.keys()):
            ## PLOT COMPARISON NUMBER SIM REAL ##
            a = np.array([np.array(self.sim_dataframe[barr][2:].to_numpy())[x] for x in range(len(self.df_barriers[barr][:-2].to_numpy())) if not self.df_barriers[barr][:-2].to_numpy()[x] == 0])
            b = np.array([np.array(self.df_barriers[barr][:-2].to_numpy())[x] for x in range(len(self.df_barriers[barr][:-2].to_numpy())) if not self.df_barriers[barr][:-2].to_numpy()[x] == 0])
            a_scaled = a/sum(a)
            b_scaled = b/sum(b)
            similarity = sum((a - b)/ b) # Percentage of difference with respect to the measured data
            similarity_scaled = sum((a_scaled - b_scaled)/ b_scaled)
            diff = (np.array(self.sim_dataframe[barr][2:].to_numpy() - self.df_barriers[barr][:-2].to_numpy())).mean()
            diff_scaled = np.array(a[2:]-b[:-2]).mean()
# AVG DISTANCE among curve same barriers real sim
            dict_avg_diff[barr] = diff
            dict_avg_diff_scaled[barr] = diff_scaled
# SIMILARITY MEASURE among same barrier sim-real #            
            dict_similarity[barr] = similarity
            dict_similarity_scaled[barr] = similarity_scaled
# avg similarity to partition barriers            
        avg_sim =np.absolute(np.array(list(dict_similarity.values()))).mean()
        avg_sim_scaled =np.absolute(np.array(list(dict_similarity_scaled.values()))).mean()
#######################
### PLOT NON SCALED ###
#######################
        for barr in list(dict_corr_same_barrier.keys()):            
 #           print('barr',barr,dict_corr_same_barrier[barr])
 #           print('barr',dict_similarity[barr],avg_sim)
            if dict_corr_same_barrier[barr]<avg and np.absolute(dict_similarity[barr])>avg_sim :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:])
                plt.plot(h,self.df_barriers[barr][:-2])
                plt.xlabel('time')
                plt.ylabel('flux')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier[barr],2),round(dict_similarity[barr],2),round(dict_avg_diff[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr + '{}/'.format('bad_corr_bad_similarity')):
                    os.mkdir(self.dir_corr + '{}/'.format('bad_corr_bad_similarity'))
                plt.savefig(self.dir_corr + '{}/'.format('bad_corr_bad_similarity') + 'comparison_bad_correlation_bad_sim_{}.png'.format(barr))
                self.list_bad_corr.append(barr)
                self.list_bad_corr_bad_sim.append(barr)
            elif dict_corr_same_barrier[barr]<avg and np.absolute(dict_similarity[barr])<avg_sim :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:])
                plt.plot(h,self.df_barriers[barr][:-2])
                plt.xlabel('time')
                plt.ylabel('flux')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier[barr],2),round(dict_similarity[barr],2),round(dict_avg_diff[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr + '{}/'.format('bad_corr_good_similarity')):
                    os.mkdir(self.dir_corr + '{}/'.format('bad_corr_good_similarity'))
                plt.savefig(self.dir_corr + '{}/'.format('bad_corr_good_similarity') + 'comparison_bad_correlation_good_sim_{}.png'.format(barr))
                self.list_bad_corr.append(barr)
                self.list_bad_corr_good_sim.append(barr)
        for barr in list(dict_corr_same_barrier.keys()):
            if dict_corr_same_barrier[barr]>avg and np.absolute(dict_similarity[barr])>avg_sim :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:])
                plt.plot(h,self.df_barriers[barr][:-2])
                plt.xlabel('time')
                plt.ylabel('flux')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier[barr],2),round(dict_similarity[barr],2),round(dict_avg_diff[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr + '{}/'.format('good_corr_bad_sim')):
                    os.mkdir(self.dir_corr + '{}/'.format('good_corr_bad_sim'))
                plt.savefig(self.dir_corr + '{}/'.format('good_corr_bad_sim') + 'comparison_good_correlations_bad_sim{}.png'.format(barr))     
                self.list_good_corr_scaled.append(barr)
                self.list_good_corr_bad_sim_scaled.append(barr)
   
            elif dict_corr_same_barrier[barr]>avg and np.absolute(dict_similarity[barr])<avg_sim :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:])
                plt.plot(h,self.df_barriers[barr][:-2])
                plt.xlabel('time')
                plt.ylabel('flux')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier[barr],2),round(dict_similarity[barr],2),round(dict_avg_diff[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr + '{}/'.format('good_corr_good_sim')):
                    os.mkdir(self.dir_corr + '{}/'.format('good_corr_good_sim'))
                plt.savefig(self.dir_corr + '{}/'.format('good_corr_good_sim') + 'comparison_good_correlations_bad_sim{}.png'.format(barr))  
                self.list_good_corr_scaled.append(barr)
                self.list_good_corr_good_sim_scaled.append(barr)
##########################
### RESCALED BARRIERS ####
##########################
        for barr in list(dict_corr_same_barrier_scaled.keys()):            
 #           print('barr',barr,dict_corr_same_barrier[barr])
 #           print('barr',dict_similarity[barr],avg_sim)
            if dict_corr_same_barrier_scaled[barr]<avg_scaled and np.absolute(dict_similarity_scaled[barr])>avg_sim_scaled :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:]/sum(self.sim_dataframe[barr][2:]))
                plt.plot(h,self.df_barriers[barr][:-2]/sum(self.df_barriers[barr][:-2]))
                plt.xlabel('time')
                plt.ylabel('flux normalized')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier_scaled[barr],2),round(dict_similarity_scaled[barr],2),round(dict_avg_diff_scaled[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr_scaled + '{}/'.format('bad_corr_bad_similarity_scaled')):
                    os.mkdir(self.dir_corr_scaled + '{}/'.format('bad_corr_bad_similarity_scaled'))
                plt.savefig(self.dir_corr_scaled + '{}/'.format('bad_corr_bad_similarity_scaled') + 'comparison_bad_correlation_bad_sim_scaled_{}.png'.format(barr))
                self.list_bad_corr_scaled.append(barr)
                self.list_bad_corr_bad_sim_scaled.append(barr)
            elif dict_corr_same_barrier_scaled[barr]<avg_scaled and np.absolute(dict_similarity_scaled[barr])<avg_sim_scaled :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:]/sum(self.sim_dataframe[barr][2:]))
                plt.plot(h,self.df_barriers[barr][:-2]/sum(self.df_barriers[barr][:-2]))
                plt.xlabel('time')
                plt.ylabel('flux normalized')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier_scaled[barr],2),round(dict_similarity_scaled[barr],2),round(dict_avg_diff_scaled[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr_scaled + '{}/'.format('bad_corr_good_similarity_scaled')):
                    os.mkdir(self.dir_corr_scaled + '{}/'.format('bad_corr_good_similarity_scaled'))
                plt.savefig(self.dir_corr_scaled + '{}/'.format('bad_corr_good_similarity_scaled') + 'comparison_bad_correlation_good_sim_scaled_{}.png'.format(barr))
                self.list_bad_corr_scaled.append(barr)
                self.list_bad_corr_good_sim_scaled.append(barr)
        for barr in list(dict_corr_same_barrier_scaled.keys()):
            if dict_corr_same_barrier_scaled[barr]>avg_scaled and np.absolute(dict_similarity_scaled[barr])>avg_sim_scaled :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:]/sum(self.sim_dataframe[barr][2:]))
                plt.plot(h,self.df_barriers[barr][:-2]/sum(self.df_barriers[barr][:-2]))
                plt.xlabel('time')
                plt.ylabel('flux normalized')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier_scaled[barr],2),round(dict_similarity_scaled[barr],2),round(dict_avg_diff_scaled[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr_scaled + '{}/'.format('good_corr_bad_sim_scaled')):
                    os.mkdir(self.dir_corr_scaled + '{}/'.format('good_corr_bad_sim_scaled'))
                plt.savefig(self.dir_corr_scaled + '{}/'.format('good_corr_bad_sim_scaled') + 'comparison_good_correlations_bad_sim_scaled{}.png'.format(barr))     
                self.list_good_corr_scaled.append(barr)
                self.list_good_corr_bad_sim_scaled.append(barr)
   
            elif dict_corr_same_barrier_scaled[barr]>avg_scaled and np.absolute(dict_similarity_scaled[barr])<avg_sim_scaled :
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:]/sum(self.sim_dataframe[barr][2:]))
                plt.plot(h,self.df_barriers[barr][:-2]/sum(self.df_barriers[barr][:-2]))
                plt.xlabel('time')
                plt.ylabel('flux normalized')
                plt.yticks(rotation = 30)
                plt.title('C = {0}, sim = {1}, avg_dist = {2} '.format(round(dict_corr_same_barrier_scaled[barr],2),round(dict_similarity_scaled[barr],2),round(dict_avg_diff_scaled[barr]),2) + barr)
                plt.legend(['sim','real'])
                plt.xticks(rotation = 30)
                if not os.path.exists(self.dir_corr_scaled + '{}/'.format('good_corr_good_sim_scaled')):
                    os.mkdir(self.dir_corr_scaled + '{}/'.format('good_corr_good_sim_scaled'))
                plt.savefig(self.dir_corr_scaled + '{}/'.format('good_corr_good_sim_scaled') + 'comparison_good_correlations_bad_sim_scaled{}.png'.format(barr))  
                self.list_good_corr_scaled.append(barr)
                self.list_good_corr_good_sim_scaled.append(barr)
                
        for i in range(len(self.list_qualitative_corr)):
            intrs = set(self.list_qualitative_corr[i]).intersection(set(self.list_qualitative_corr_scaled[i]))
            c = self.list_qualitative_corr[i] if len(self.list_qualitative_corr[i])>len(self.list_qualitative_corr_scaled[i]) else self.list_qualitative_corr_scaled[i] 
            print('intersection\t',intrs,'\t out of',c)

        
        print('good corr\t',self.list_good_corr,'\tbad corr\t',self.list_bad_corr)
      

#        fig = plt.figure(figsize=(30, 15))
#        plt.bar(x = list(dict_corr_same_barrier_norm.keys()),height = list(dict_corr_same_barrier_norm.values()))
#        plt.xticks(rotation = 90)
#        plt.title('Correlation same barrier sim-real data normalized')
#        plt.savefig(sim.state_basename + plots_ + 'corr_same_barrier_normalized.png')

        
        # PLOT (CORRELATIONS SIM) - (CORRELATIONS REAL) 
        self.dif_cor = (corrMatrix - corrMatrix1)**2
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(self.dif_cor)
        plt.title('difference in correlation matrix sim-real data')
        plt.savefig(sim.state_basename + plots_ + 'difference_correlation_matrices_real_sim.png')
        # PLOT SIM-REAL NORMED CORRELATIONS
        self.dif_cor_norm = (corrMatrix_norm - corrMatrix1_norm)**2
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(self.dif_cor)
        plt.title('difference in correlation matrix sim-real data normalized')
        plt.savefig(sim.state_basename + plots_ + 'difference_correlation_matrices_normalized_real_sim.png')
        # PLOT SIM CORRELATIONS
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(corrMatrix)
        plt.title('correlation matrix sim')
        plt.savefig(sim.state_basename + plots_ + 'correlation_matrices_sim.png')
        # PLOT REAL CORRELATIONS
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(corrMatrix1)
        plt.title('correlation matrix real')
        plt.savefig(sim.state_basename + plots_ + 'correlation_matrices_real.png')
        # PLOT SIM NORMALIZED CORRELATIONS
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(corrMatrix)
        plt.title('correlation matrix sim data normalized')
        plt.savefig(sim.state_basename + plots_ + 'correlation_matrices_sim_normalized.png')
        
        # PLOT SIM NORMALIZED CORRELATIONS
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(corrMatrix1)
        plt.title('correlation matrix real data normalized')
        plt.savefig(sim.state_basename + plots_ + 'correlation_matrices_real_normalized.png')


#        plt.show()
        self.cost_corr = 0.5*sum([sum(x) for x in self.dif_cor.fillna(0).to_numpy()])
        return plots_, self.cost_corr

    def shreckerberg_alg(self):
        '''Shreckerberg algorithm:
        Still need to consider the inverse partecipation.'''
        self.corr_sim = self.sim_dataframe.drop(['timestamp','datetime']).apply(lambda x:x-x.mean()).corr()
        self.corr_data = self.df_barriers.drop('timestamp').apply(lambda x: x-x.mean()).corr()
        self.corr_dif = self.corr_sim - self.corr_data
        self.barr2no = dict(zip(self.df_barriers.drop('timestamp').columns,np.arange(len(list(self.df_barriers.drop('timestamp').columns)))))
        self.U_sim,self.S_sim,self.V_sim = np.linalg.svd(self.corr_sim)
        self.U_diff,self.S_diff,self.V_diff = np.linalg.svd(self.corr_dif)
        self.U_data,self.S_data,self.V_data = np.linalg.svd(self.corr_data)
        self.inverse_partecipation =  1
    def comparison_number_people_sim_real(self,sim,plots_):
        '''I reproduce the graph with the number of people in the simulation and in the real data.'''
        if sim.average_fluxes:
#            print(sim.df_avg.drop('timestamp',axis = 1).apply(lambda x: sum(x)))
            total_number_people_real = sum(sim.df_avg.drop('timestamp',axis = 1).apply(lambda x: sum(x)))
            total_number_people_real_per_hour = [sum(row) for _, row in sim.df_avg.drop('timestamp',axis = 1).iterrows()]
            cumulative_per_hour_real = []
            c = 0
            for x in total_number_people_real_per_hour:
                c = c + x
                cumulative_per_hour_real.append(c) 
            total_number_people_sim = sum(self.sim_dataframe.drop(['timestamp','datetime'],axis = 1).apply(lambda x:sum(x)))
            total_number_people_sim_per_hour = [sum(row) for _, row in self.sim_dataframe.drop(['timestamp','datetime'],axis = 1).iterrows()]
            cumulative_per_hour_sim = []
            c = 0
            for x in total_number_people_sim_per_hour:
                c = c + x
                cumulative_per_hour_sim.append(c) 
            fig = plt.figure(figsize=(30, 15))
            plt.plot(sim.df_avg['timestamp'],total_number_people_real_per_hour)
            plt.plot(sim.df_avg['timestamp'],total_number_people_sim_per_hour)
            plt.xlabel('time (h)')
            plt.ylabel('number of people')
            plt.legend(['real','sim'])
            plt.title('total number in sim {0}, real {1}'.format(total_number_people_sim,total_number_people_real))
            plt.savefig(sim.state_basename + plots_ + 'comparison_number_people_per_hour_sim_real.png')
            fig = plt.figure(figsize=(30, 15))
            plt.plot(sim.df_avg['timestamp'],cumulative_per_hour_real)
            plt.plot(sim.df_avg['timestamp'],cumulative_per_hour_sim)
            plt.xlabel('time (h)')
            plt.ylabel('cumulative number of people')
            plt.legend(['real','sim'])
            plt.title('total number in sim {0}, real {1}'.format(total_number_people_sim,total_number_people_real))
            plt.savefig(sim.state_basename + plots_ + 'comparison_cumulative_number_people_per_hour_sim_real.png')
#            print('difference between pawns real sim',np.array(cumulative_per_hour_real) - np.array(cumulative_per_hour_sim))
        if sim.pick_day:
            total_number_people_real = sum(sim.df_day.drop('timestamp',axis = 1).apply(lambda x: sum(x)))
            total_number_people_real_per_hour = [sum(row) for _, row in sim.df_day.drop('timestamp',axis = 1).iterrows()]
            cumulative_per_hour_real = []
            c = 0
            for x in total_number_people_real_per_hour:
                c = c + x
                cumulative_per_hour_real.append(c) 
            total_number_people_sim = sum(self.sim_dataframe.drop(['timestamp','datetime'],axis = 1).apply(lambda x:sum(x)))
            total_number_people_sim_per_hour = [sum(row) for _, row in self.sim_dataframe.drop(['timestamp','datetime'],axis = 1).iterrows()]
            cumulative_per_hour_sim = []
            c = 0
            for x in total_number_people_sim_per_hour:
                c = c + x
                cumulative_per_hour_sim.append(c) 
            fig = plt.figure(figsize=(30, 15))
            plt.plot(sim.df_day['timestamp'],total_number_people_real_per_hour)
            plt.plot(sim.df_day['timestamp'],total_number_people_sim_per_hour)
            plt.xlabel('time (h)')
            plt.ylabel('number of people')
            plt.legend(['real','sim'])
            plt.title('total number in sim {0}, real {1}'.format(total_number_people_sim,total_number_people_real))
            plt.savefig(sim.state_basename + plots_ + 'comparison_number_people_per_hour_sim_real.png')
            fig = plt.figure(figsize=(30, 15))
            plt.plot(sim.df_day['timestamp'],cumulative_per_hour_real)
            plt.plot(sim.df_day['timestamp'],cumulative_per_hour_sim)
            plt.xlabel('time (h)')
            plt.ylabel('cumulativenumber of people')
            plt.legend(['real','sim'])
            plt.title('total number in sim {0}, real {1}'.format(total_number_people_sim,total_number_people_real))
            plt.savefig(sim.state_basename + plots_ + 'comparison_cumulative_number_people_per_hour_sim_real.png')
            print('difference between pawns real sim for the day',np.array(cumulative_per_hour_real) - np.array(cumulative_per_hour_sim))
         
            
    def focus_comparison_fluxes(self,sim,plots_):
        '''Compares the fluxes of barriers corresponding to sources and other specific barriers:
        Essentially those of some streets of interest (STRADA NOVA, STRADA PRINCIPALE)
        In this case I expect 5*4 plots (1+4)*4'''
        legend_ = ['sim','real']
        self.list_src = ['COSTITUZIONE_1_IN','COSTITUZIONE_1_OUT','PAPADOPOLI_1_IN','PAPADOPOLI_1_OUT','SCALZI_2_IN','SCALZI_2_OUT','SCALZI_3_IN','SCALZI_3_OUT']
        self.list_strada_nova = ['MADDALENA_1_IN','MADDALENA_1_OUT','FARSETTI_1_IN','FARSETTI_1_OUT','SANFELICE_1_IN','SANFELICE_1_OUT','SANTASOFIA_1_IN','SANTASOFIA_1_OUT']#['SCALZI_2_IN','SCALZI_3_IN','MADDALENA_1_IN','FARSETTI_1_IN','SANFELICE_1_IN','SANTASOFIA_1_IN','PISTOR_1_IN','APOSTOLI_1_IN','GRISOSTOMO_1_IN']
        self.list_strada_principale = ['SANGIACOMO_1_IN','SANGIACOMO_1_OUT','SANAGOSTIN_1_IN','SANAGOSTIN_1_OUT','TERAANTONIO_1_IN','TERAANTONIO_1_OUT','MADONETA_1_IN','MADONETA_1_OUT']#['SCALZI_2_IN','SCALZI_3_IN','SANGIACOMO_1_IN','SANAGOSTIN_1_IN','TERAANTONIO_1_IN','MADONETA_1_IN','MANDOLA_2_IN']
        self.list_strada_occ = ['TREPONTI_1_IN','TREPONTI_1_OUT','RAGUSEI_1_IN','RAGUSEI_1_OUT','RAGUSEI_2_IN','RAGUSEI_2_OUT','RAGUSEI_3_IN','RAGUSEI_3_OUT']#['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','TREPONTI_1_IN','RAGUSEI_1_IN','RAGUSEI_2_IN','RAGUSEI_3_IN','RAGUSEI_4_IN','BARNABA_1_IN','CASINNOBILI_1_IN']
        self.list_san_rocco = ['CALLELACA_1_IN','CALLELACA_1_OUT','SANROCCO_1_IN','SANROCCO_1_OUT']
        self.list_pescheria = ['SOTBETTINA_1_IN','SOTBETTINA_1_OUT','SOTCAPELER_1_IN','SOTCAPELER_1_OUT']
        self.list_all = [self.list_src,self.list_strada_nova,self.list_strada_principale,self.list_strada_occ,self.list_san_rocco,self.list_pescheria]
        self.labels_streets = ['sources','strada nova','strada to San Marco','West street','San Rocco','Pescheria'] 
        self.dict_percentage_mobility_in = {'sources':{'sim': 0, 'real':0},'strada nova':{'sim': 0, 'real':0},'strada to San Marco':{'sim': 0, 'real':0},'West street':{'sim': 0, 'real':0},'San Rocco':{'sim': 0, 'real':0},'Pescheria':{'sim': 0, 'real':0}}
        self.dict_percentage_mobility_out = {'sources':{'sim': 0, 'real':0},'strada nova':{'sim': 0, 'real':0},'strada to San Marco':{'sim': 0, 'real':0},'West street':{'sim': 0, 'real':0},'San Rocco':{'sim': 0, 'real':0},'Pescheria':{'sim': 0, 'real':0}} 
        h = list(np.arange(22))
        c = 0
        total_count_in ={'sim':0,'real':0}
        total_count_out = {'sim':0,'real':0}
        for list_ in self.list_all:
            ## CORRELATION ##
            i = 0
            fig = plt.figure()
            corrMatrix = self.sim_dataframe[list_].corr()
            sns.heatmap(corrMatrix ,annot = True,cbar = True)
            plt.title('Correlation Matrix sim {0} {1}'.format(sim.start_date,self.labels_streets[c]))
            plt.xticks(rotation = 30)
            if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
                os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
            plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) +'correlation_matrix_sim_{}.png'.format(self.labels_streets[c]),dpi = 250)
#                plt.show()
            for barr in list_:
                ## PLOT COMPARISON NUMBER SIM REAL ##
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:])
                plt.plot(h,self.df_barriers[barr][:-2])
                plt.xlabel('time')
                plt.ylabel('flux')
                plt.yticks(rotation = 30)
                plt.title(list_[i] + ' ' + sim.start_date)
                plt.legend(legend_)
                plt.xticks(rotation = 30)

                if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
                    os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
                plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) + 'comparison_per_hour_sim_real_{}.png'.format(barr))
#                    plt.show()
### RESCALED CASE ###
                fig = plt.figure()
                plt.plot(h,self.sim_dataframe[barr][2:]/sum(self.sim_dataframe[barr][2:]))
                plt.plot(h,self.df_barriers[barr][:-2]/sum(self.df_barriers[barr][:-2]))
                plt.xlabel('time')
                plt.ylabel('flux scaled')
                plt.yticks(rotation = 30)
                plt.title(list_[i] + ' ' + sim.start_date)
                plt.legend(legend_)
                plt.xticks(rotation = 30)

                
                if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
                    os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
                plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) + 'scaled_comparison_per_hour_sim_real_{}.png'.format(barr))
                i = i + 1
                if '_IN' in barr:
                    total_count_in['sim'] += sum(self.sim_dataframe[barr][2:])
                    total_count_in['real'] += sum(self.df_barriers[barr][:-2])
                else:
                    total_count_out['sim'] += sum(self.sim_dataframe[barr][2:])                    
                    total_count_out['real'] += sum(self.df_barriers[barr][:-2])
            for list_ in self.list_all:
                for barr in list_:
                    if '_IN' in barr:
                        self.dict_percentage_mobility_in[self.labels_streets[c]]['sim'] += sum(self.sim_dataframe[barr][2:]/total_count_in['sim'])
                        self.dict_percentage_mobility_in[self.labels_streets[c]]['real'] += sum(self.df_barriers[barr][:-2]/total_count_in['real'])
                    else:
                        self.dict_percentage_mobility_out[self.labels_streets[c]]['sim'] += sum(self.sim_dataframe[barr][2:]/total_count_out['sim']) 
                        self.dict_percentage_mobility_out[self.labels_streets[c]]['real'] += sum(self.df_barriers[barr][:-2]/total_count_out['real'])

#                    plt.show()
            c = c + 1
            
#        fig = plt.figure()
#        plt.plot(h,self.sim_dataframe['SCALZI_2_OUT'][:-2])
#        plt.plot(h,self.df_barriers['SCALZI_2_OUT'][:-2])
#        plt.xlabel('time')
#        plt.ylabel('flux')
#        plt.yticks(rotation = 30)
#        plt.title('SCALZI_2_OUT' + ' ' + sim.start_date)
#        plt.legend(legend_)
#        plt.xticks(rotation = 30)
#        plt.savefig(sim.state_basename + plots_ +'SCALZI_2_OUT.png' )
#        fig = plt.figure()
#        plt.plot(h,self.sim_dataframe['SCALZI_3_OUT'][:-2])
#        plt.plot(h,self.df_barriers['SCALZI_3_OUT'][:-2])
#        plt.xlabel('time')
#        plt.ylabel('flux')
#        plt.yticks(rotation = 30)
#        plt.title('SCALZI_3_OUT' + ' ' + sim.start_date)
#        plt.legend(legend_)
#        plt.xticks(rotation = 30)
#        plt.savefig(sim.state_basename + plots_ +'SCALZI_3_OUT.png' )
    
        
    
    def comparison_corr_subsets(self,sim,plots_):
        '''Description:
        Calculates the best delayed correlation functions between in and out barriers for all the different routes.
        Plot them.'''   
        deltat = np.arange(0,6)
        cols = []
        for l in self.list_all:
            for l0 in l:
                if '_IN' in l0:
                    cols.append(l0)
        cols.append('deltat')
        df_ret_real = dict.fromkeys(cols)
        for k in df_ret_real.keys():
            df_ret_real[k] = []
#        df_ret_real_norm = dict.fromkeys(cols)
#        self.creation_default_dict(df_ret_real_norm)
        df_ret_sim = dict.fromkeys(cols)
        for k in df_ret_sim.keys():
            df_ret_sim[k] = []

#        df_ret_sim_norm = dict.fromkeys(cols) 
#        self.creation_default_dict(df_ret_sim_norm)       
        for l in self.list_all:
            for l0 in l:
                for l1 in l:
                    if l1 != l0 and l1.split('_')[0] + '_' +l1.split('_')[1] in l0 and '_IN' in l0 and not '_IN' in l1:                        
                        for t in deltat:
#                            print('append in \t',l0,'\tl1',l1)
                            df_ret_real[l0].append(self.pearson_corr(np.roll(np.array(self.tmp_flux_real.drop('timestamp',axis = 1)[l0]),int(t))[int(t):],np.array(self.tmp_flux_real.drop('timestamp',axis = 1)[l1])[0:len(self.tmp_flux_real)-int(t)]))
#                            df_ret_real_norm[l0].append(self.pearson_corr(self.tmp_flux_real_norm[l0],self.tmp_flux_real_norm[l1]))
#                            df_ret_real_norm['time'].append(t)
                            df_ret_sim[l0].append(self.pearson_corr(np.roll(np.array(self.tmp_flux_sim.drop('timestamp',axis = 1)[l0]),int(t))[int(t):len(self.tmp_flux_sim)],np.array(self.tmp_flux_sim.drop('timestamp',axis = 1)[l1])[0:len(self.tmp_flux_sim)-int(t)]))
#                            df_ret_sim_norm[l0].append(self.pearson_corr(self.tmp_flux_sim_norm[l0],self.tmp_flux_sim_norm[l1]))
#                            df_ret_sim_norm['time'].append(t)
                    else:
                        pass
#        print(df_ret_real)
        df_ret_real['deltat'] = list(deltat)
        df_ret_sim['deltat'] = list(deltat)
        df_ret_real = pd.DataFrame(df_ret_real)
        df_ret_real = df_ret_real.set_index('deltat')
#        df_ret_real_norm = pd.DataFrame(df_ret_real_norm,index = list(deltat))
        df_ret_sim = pd.DataFrame(df_ret_sim)
        df_ret_sim = df_ret_sim.set_index('deltat')       
# df_ret_sim_norm = pd.DataFrame(df_ret_sim_norm,index = list(deltat))
        c = 0
        for list_ in self.list_all:
            fig = plt.figure(30,15)
            ax = df_ret_real.plot.bar(rot=90,stacked = True)
            if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
                os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
            plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) + 'corr_in_out_real.png')

#            fig = plt.figure(30,15)
#            ax = df_ret_real_norm.plot.bar(rot=90)
#            if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
#                os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
#            plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) + 'corr_in_out_norm.png')

            fig = plt.figure(30,15)
            ax = df_ret_sim.plot.bar(rot=90,stacked = True)
            if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
                os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
            plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) + 'corr_in_out_sim.png')
#            fig = plt.figure(30,15)
#            ax = df_ret_sim_norm.plot.bar(rot=90)
#            if not os.path.exists(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c])):
#                os.mkdir(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]))
#            plt.savefig(sim.state_basename + plots_ + '{}/'.format(self.labels_streets[c]) + 'corr_in_out_sim_norm.png')
#                    plt.show()
            c = c + 1
                
        return 0


    def compare_cumulative_fluxes(self,sim):
        '''Input:
        self: sim_dataframe, df_barriers 
        sim: state_basename
        -----------------------------
        Description:
        
        Create dictionary_idx_name: keys: name_barrier_real, values increasing number. [it will be used for correlation matrices]
        Control that cumulative of some barriers are less or equal to the cumulative of other barriers
        that seem to be the principal contributors for the fluxes there.
        TODO: create dict_difference: keys ->  barriera -> values indeces barriera whose sum give barriera
        Compare them to waiting times (1000)
        Plot those that are chosen 
         '''
        self.dict_idx_name_real = dict.fromkeys(self.df_barriers.drop(columns = ['timestamp']).columns)
        self.dict_idx_name_sim = dict.fromkeys(self.sim_dataframe.drop(columns = ['timestamp','datetime']).columns)
        c = 0
        for k in self.dict_idx_name_real.keys():
            self.dict_idx_name_real[k] = c
            c = c + 1 
        c = 0
        for k in self.dict_idx_name_sim.keys():
            self.dict_idx_name_real[k] = c
            c = c + 1    
        list_to_compare = ['PAPADOPOLI_1_IN','SCALZI_2_IN','SCALZI_2_OUT','SCALZI_3_IN','SCALZI_3_OUT','COSTITUZIONE_1_IN']
        time_interval_comparison = 3
        self.cumulative_it = np.cumsum(self.sim_dataframe.drop(columns = ['timestamp','datetime'])[['PAPADOPOLI_1_IN','SCALZI_2_IN','SCALZI_2_OUT','SCALZI_3_IN','SCALZI_3_OUT','COSTITUZIONE_1_IN']].to_numpy(dtype = int),axis = 0)
 #       print('shape cumulative',np.shape(self.cumulative_it))
        a = pd.DataFrame(self.cumulative_it,columns = list_to_compare )
        waiting_time = (self.cumulative_it[:,2] + self.cumulative_it[:,4])/1000 - 1
        people_home = self.cumulative_it[:,1] + self.cumulative_it[:,3]
        total_people_in = self.cumulative_it[:,0] + self.cumulative_it[:,5]
        a['waiting_time'] = waiting_time
        a['people_home'] = people_home 
        a['total_people_in'] = total_people_in 
#        print(a)
        a.to_csv(sim.state_basename + '/cumulative.csv',';')
        for k in range(len(self.cumulative_it)-3):
#            print('difference in cumulative in scalzi_2 in e out time {0}-{1}'.format(str(k),str(k+1)), self.cumulative_it[k][2] - self.cumulative_it[k+1][1])
            print('difference in cumulative in scalzi_2 in e out time {0}-{1}'.format(str(k),str(k+2)), self.cumulative_it[k][2] - self.cumulative_it[k+2][1])
            print('{0}-{1}'.format(list_to_compare[2],list_to_compare[1]),'time {0}-{1}'.format(str(k),str(k+3)), self.cumulative_it[k][2] - self.cumulative_it[k+3][1])         
                     
                     
#        self.difference_cumulative = np.zeros(len(self.cumulative_it),len(self.cumulative_it[0]))
#        for j in range(len(self.cumulative_it[0])):
#            self.difference_cumulative[:,j] = np.array(self.cumulative_it - self.cumulative_it[:,j])

    #### NORMALIZE AND CORRELATION MATRICES ####
    def normalize_fluxes(self):
        '''Normalizes fluxes and creates df_norm_fluxes
        That has columns (timestamp,BARRIER_IN,BARRIER_OUT)
        timestamp contains just the hours if averaged'''
        self.df_norm_fluxes_sim = self.sim_dataframe.drop(columns = ['timestamp','datetime']).apply(lambda x: x/sum(x))
        self.df_norm_fluxes_sim['datetime'] = self.sim_dataframe['datetime']



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
        self.std_correlation_matrix_real_data = self.sim_dataframe.drop(columns = ['datetime','timestamp']).apply(lambda x: (x - x.mean())/np.std(x)) 
        

    def initialize_A(self):
        '''Input:
        ------------------------
        Object simulation:
        Is the matrix from Schreckenberg paper for which I apply the singular value decomposition'''
        self.A_real_data = self.sim_dataframe.drop(columns = ['datetime','timestamp']).apply(lambda x: x - x.mean()) 
    
    def ward_plot(self,sim,dendogram_):
        '''Input:
        -----------------
        Simulation object: ->
        Description:
        ------------------
        Plots the ward dendogram and produces the distance between real and sim ward graph both symmetric and no.
        It saves these dictionaries whose keys are ['tag1-tag2'] for element in tagn, and values are the 
        diagrammatic distance.
        self.symmetric_ward_distance_sim_real,  
        self.symmetric_ward_distance_sim_real_norm,
        self.ward_distance_sim_real, 
        self.ward_distance_sim_real are the output of interest
        self.dict_list_cluster: contains for each situation (sim_(,norm),real_(,norm) the number of the cluster the sim.tags stays on)
        self.dict_ncluster: contains for each situation (sim_(,norm),real_(,norm) the number of clusters in total: must be less then number colors)
        
        Output:
        symmetric_ward_distance_sim_real: dict.keys() = couples of barriers
        

        '''
        self.dist_type = 'euclidean'
        self.hca_method = 'ward'
        self.nan_t = 5 # %. droppa le barriers con troppi nan
        self.sm_window = 9 # smoothing window, deve essere dispari
        self.sm_order = 3 # smoothing polynomial order
        self.iconcolors = ['purple', 'orange', 'red', 'green', 'blue', 'pink', 'beige', 'darkred', 'darkpurple', 'lightblue', 'lightgreen', 'cadetblue', 'lightgray', 'gray', 'darkgreen', 'white', 'darkblue', 'lightred', 'black']
        self.dict_distance_ward = dict.fromkeys(list(self.dict_df_ward.keys()))
        self.dict_symmetric_distance_ward = dict.fromkeys(list(self.dict_df_ward.keys()))
        self.dict_list_cluster = dict.fromkeys(list(self.dict_df_ward.keys()))
        self.dict_ncluster = dict.fromkeys(list(self.dict_df_ward.keys()))
        
        print('ward plot')
        if self.windows:
            if self.n_epoch == 0:
                dendogram_ = '/dendogram_plots_{}/'.format(0)
            else:
                dendogram_ = dendogram_ + '/dendogram_plots_{}/'.format(self.n_epoch)
        else:
            if self.n_epoch == 0:
                dendogram_ = '\\dendogram_plots_{}\\'.format(0)
            else:
                dendogram_ = dendogram_ + '\\dendogram_plots_{}\\'.format(self.n_epoch)
        if not os.path.exists(sim.state_basename + dendogram_):
            os.mkdir(sim.state_basename + dendogram_)        
        for k in list(self.dict_df_ward.keys()):
            print('key dictionary ward',k)
            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot(111)
            try:
                print('clustering')
                linkg = sch.linkage(self.dict_df_ward[k]['distance'].to_numpy(),method=self.hca_method)#,optimal_ordering = True     
                self.dict_distance_ward[k], self.dict_symmetric_distance_ward[k] = self.calculate_distance_ward(linkg, sim.tags)
                ddata = sch.dendrogram(linkg,labels=sim.tags) 
                if 'sim' in k:
                    self.cut_distance = 0.28 # distanza nel dendrogramma a cui tagliare per decidere il numero di cluster
                else:
                    self.cut_distance = 0.3 # distanza nel dendrogramma a cui tagliare per decidere il numero di cluster                    
                for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                    x = 0.5 * sum(i[1:3])
                    y = d[1]
#                if y > self.cut_distance:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                            textcoords='offset points',
                            va='top', ha='center')
                if self.cut_distance:
                    plt.axhline(y= self.cut_distance, c='k')

                clusterlist = sch.fcluster(linkg,self.cut_distance, criterion='distance')-1 
                self.dict_list_cluster[k] = pd.DataFrame({'barrier':sim.tags,'cluster':clusterlist})
                self.dict_ncluster[k] = len(np.unique(self.dict_list_cluster[k]['cluster']))
                assert self.dict_ncluster[k] <= len(self.iconcolors) # troppi pochi colori per tutti i cluster altrimenti
            except:
                pass
            plt.title('HCA dendrogram method: {0}, case: {1}'.format(self.hca_method,k.split('_')[1] + k.split('_')[2]))
            plt.xticks(fontsize=7)
            plt.savefig(sim.state_basename + dendogram_ + f'HCA dendrogram method_{self.hca_method}_case_{k}.png')
#            plt.show()
        self.symmetric_ward_distance_sim_real = 0
        self.symmetric_ward_distance_sim_real_norm = 0
        self.ward_distance_sim_real = 0
        self.ward_distance_sim_real_norm = 0
        for k in list(self.dict_df_ward.keys()):
            for k1 in list(self.dict_df_ward.keys()):
                if k == 'df_dist_sim' and k1 == 'df_dist_real':
                    for key in list(self.dict_symmetric_distance_ward[k]):
                        self.symmetric_ward_distance_sim_real = self.symmetric_ward_distance_sim_real + self.dict_symmetric_distance_ward[k][key]
                        self.ward_distance_sim_real = self.ward_distance_sim_real  + self.dict_distance_ward[k][key]
                    self.symmetric_ward_distance_sim_real = self.symmetric_ward_distance_sim_real/2
                    self.ward_distance_sim_real = self.ward_distance_sim_real/2
                try:
                    if k == 'df_dist_sim_norm' and k1 == 'df_dist_real_norm':
                        for key in list(self.dict_symmetric_distance_ward[k]):
                            self.symmetric_ward_distance_sim_real_norm = self.symmetric_ward_distance_sim_real_norm + self.dict_symmetric_distance_ward[k][key]
                            self.ward_distance_sim_real = self.ward_distance_sim_real + self.dict_symmetric_distance_ward[k][key]
                        self.symmetric_ward_distance_sim_real_norm = self.symmetric_ward_distance_sim_real_norm/2
                        self.ward_distance_sim_real_norm = self.ward_distance_sim_real_norm/2
                except:
                    print('colud not calculate ward distance for normalized data. It happense if I pick day, probably for it exists a set of sources that are not present always all the week ')
        return dendogram_,self.symmetric_ward_distance_sim_real,self.symmetric_ward_distance_sim_real_norm,self.ward_distance_sim_real,self.ward_distance_sim_real_norm

    def calculate_distance_ward(self,z,labels):
        '''Input:
        ----------------------
        z: list: np.shape(z) = [len(labels)-1,4] is linkg from sch.linkg
        labels: list -> ordered list of barriers
        Output:
        ----------------------
        dict_symmetric_distance 
        dict_distance: dict -> keys (couples of indices 'k0-k1' of labels that are mapped in the name of the barriers)
        values : distance between k0,k1 intended as the number of steps to go up to the first common descendent
        '''
        n = len(labels)
        l = list(np.arange(n))                                     
        dict_binary = dict.fromkeys(l)
        for k in list(dict_binary.keys()):
            dict_binary[k] = [k]
        for key,value in dict_binary.items():
            list_control = []
            for v in value:
                for i in range(np.shape(z)[0]):
                    if z[i,0] == v or z[i,1] == v and v not in list_control:
                        value.append(n+i)
                        if v not in list_control:
                            list_control.append(v)
        ### CALCULATING DISTANCE AND SYMMETRIC DISTANCE ###
        list_length_cluster_path = []
        for k in list(dict_binary.keys()):
            list_length_cluster_path.append(len(dict_binary[k]))
        max_depth = max(list_length_cluster_path)

#        print(dict_binary[0][1:-1])
        list_couple_keys = []
        for k0 in list(dict_binary.keys()):
            for k1 in list(dict_binary.keys()):
                list_couple_keys.append(str(k0)+'-'+str(k1))
        dict_distance  = dict.fromkeys(list_couple_keys)
        dict_symmetric_distance = dict.fromkeys(list_couple_keys)
        for k0 in list(dict_binary.keys()):
            for k1 in list(dict_binary.keys()):
                c = 0
                for v in dict_binary[k0][1:]:
                    if v in dict_binary[k1][1:]:
                        dict_distance[str(k0)+'-'+str(k1)] = c
                        d = 0
                        for v1 in  dict_binary[k1][1:]:
                            if v1 in dict_binary[k0][1:]:
                                dict_symmetric_distance[str(k0 )+'-'+str(k1)] = max([c,d])
                                break
                            d = d + 1
                    c = c +1


        ward_distance = 0 
        for key in list(dict_symmetric_distance.keys()):
            ward_distance = ward_distance + dict_symmetric_distance[key]
        ward_distance = ward_distance/2

        return dict_symmetric_distance,dict_distance

    def plot_cluster_real_behavior(self,sim,plots_):
        '''Looking at the clustering for real data'''
        import time
        h = list(np.arange(22))
        print('starting to plot comparison')
        for k in list(self.dict_df_ward.keys()):
            print(k)
            if k != 'df_dist_real_norm' and k != 'df_dist_sim_norm':
                list_control = []
                grp = self.dict_list_cluster[k].groupby('cluster')
                for clst ,dfb in grp:
                    print('cluster number',clst)
                    t0 = time.time()
                    for b1 in dfb['barrier']:
                        if not '_OUT' in b1 and not (b1 in list_control):
                            list_control.append(b1)
                            for b2 in dfb['barrier']:
                                if not '_OUT' in b2 and not (b2 in list_control):
                                    if b1 != b2:
                                        try:
                                            fig = plt.figure()
                                            plt.plot(h,self.sim_dataframe[b1][2:])
                                            plt.plot(h,self.df_barriers[b2][:-2])
                                            plt.xlabel('time')
                                            plt.ylabel('flux')
                                            plt.title('comparison' + b1 +' ' + b2)
                                            plt.legend([b1,b2])
                                            if not os.path.exists(sim.state_basename + plots_ + 'comparison_streets_same_cluster_{0}_{1}/'.format(clst,k.split('_')[2])):
                                                os.mkdir(sim.state_basename + plots_ + 'comparison_streets_same_cluster_{0}_{1}/'.format(clst,k.split('_')[2]))
                                            plt.savefig(sim.state_basename + plots_ + 'comparison_streets_same_cluster_{0}_{1}/'.format(clst,k.split('_')[2]) + '{0}_{1}.png'.format(b1,b2))
                                        except KeyError:
                                            print(b1,'or',b2,' are not in the keys') 
                    t1 = time.time()
                    print('time to save cluster\t',clst,'\t is \t',t1-t0)
    



class barrier:
    '''Barrier class:
    -----------------
    Input:
    name: str -> APOSTOLI_1_IN
    VARIABLES DESCRIPTION:
    flux: flux[name_barrier] = flux_name_barrier[time] = flux
    time comes from analyzer.sim_dataframe['datetime']
    name_barrier comes from analyzer.sim_dataframe.columns
    '''
    def __init__(self):
        self.name = ''
        self.flux = {}
