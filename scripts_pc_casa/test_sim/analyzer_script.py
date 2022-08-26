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
    from setting_functions_sim import *
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
    '''
    def __init__(self,
                 sim_):
        self.dir_plot = sim_.dir_plot
        self.sim_date = sim_.start_date
        self.df_barriers = sim_.data_barriers
        self.str0 = sim_.start_date.split('-')[0] + sim_.start_date.split('-')[1] + sim_.start_date.split('-')[2].split(' ')[0] 
        self.str1 = sim_.start_date.split('-')[2].split(' ')[1].split(':')[0] +sim_.start_date.split('-')[2].split(' ')[1].split(':')[1] + sim_.start_date.split('-')[2].split(' ')[1].split(':')[2]
        self.sim_dataframe = pd.read_csv(sim_.state_basename+'\\venezia_barriers_{}_{}.csv'.format(self.str0,self.str1)) 

        if sim_.pick_day:
            self.df = sim_.df_day            
        else:
            self.df = sim_.df_avg
        
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
        
        '''
        str0 = sim_.start_date.split('-')[0] + sim_.start_date.split('-')[1] + sim_.start_date.split('-')[2].split(' ')[0] 
        str1 = sim_.start_date.split('-')[2].split(' ')[1].split(':')[0] +sim_.start_date.split('-')[2].split(' ')[1].split(':')[1] + sim_.start_date.split('-')[2].split(' ')[1].split(':')[2]
        self.sim_dataframe = pd.read_csv(sim_.state_basename+'\\venezia_barriers_{}_{}.csv'.format(sim_.str0,sim_.str1)) 
        if sim_.pick_day:
            self.df_barriers = sim_.df_day            
        else:
            self.df_barriers = sim_.df_avg
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
            a = k.split('_')[0]
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

    #### NORMALIZE AND CORRELATION MATRICES ####
    def normalize_fluxes(self):
        '''Normalizes fluxes and creates df_norm_fluxes
        That has columns (timestamp,BARRIER_IN,BARRIER_OUT)
        timestamp contains just the hours if averaged'''
        self.df_norm_fluxes_sim = self.sim_dataframe.drop(columns = ['timestamp','datetime']).apply(lambda x: x/sum(x))
        self.df_norm_fluxes_sim = self.sim_dataframe['datetime']



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
            
    
    def distance_csv_for_ward(self,sim):
        '''Input:
        ---------------------
        sim: simulation object
        Description: saves df_dist_sim,df_norm_dist_sim, df_dist_real and df_norm_dist_real -> columns ['tag1','tag2','barrier1','barrier2','day','distance']
        They contain the Ward data for 1 day or the averaged period depending on the simulation.
        Here there is an allineation between simulation and real data.
        self.dict_df_ward contains the files [barrier1,barrier2,tag1,tag2,distance] for each simulation and real data,
        both normalized and no'''
        self.normalize_fluxes()
        t_dist_real = []
        t_dist_real_norm = []
        t_dist_sim = []
        t_dist_sim_norm = []
        for j in range(0,tagn-1):
            for k in range(j+1,tagn):
                dist_sim = dist(norm(self.sim_dataframe[tags[j]].values),norm(self.sim_dataframe[tags[k]].values),dist_type)
                norm_dist_sim = dist(norm(self.df_norm_fluxes_sim[tags[j]].values),norm(self.df_norm_fluxes_sim[tags[k]].values),dist_type)                 
                if sim.average_fluxes:
                    norm_dist_real = dist(norm(sim.df_norm_fluxes[tags[j]].values),norm(sim.df_norm_fluxes[tags[k]].values),dist_type)                 
                    dist_real = dist(norm(sim.df_avg[tags[j]].values),norm(sim.df_avg[tags[k]].values),dist_type)
                else:
                    norm_dist_real = dist(norm(sim.df_norm_fluxes[tags[j]].values),norm(sim.df_norm_fluxes[tags[k]].values),dist_type)                 
                    dist_real = dist(norm(sim.df_day[tags[j]].values),norm(sim.df_day[tags[k]].values),dist_type)
                t_dist_real.append([dist_real,j,k,sim.start_date])
                t_dist_real_norm.append([norm_dist_real,j,k,sim.start_date])
                t_dist_sim.append([dist_sim,j,k,sim.start_date])
                t_dist_sim_norm.append([norm_dist_sim,j,k,sim.start_date])

        ### REAL DATA ####
        t_dist_real_norm = np.array(t_dist_real_norm)
        t_dist_real = np.array(t_dist_real)    
        df_dist_real_norm = pd.DataFrame()
        df_dist_real = pd.DataFrame()
        df_dist_real_norm['tag1']= t_dist_real_norm[:,1].astype(int)
        df_dist_real_norm['tag2']= t_dist_real_norm[:,2].astype(int)#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_real_norm['barrier1'] = [tags[int(a)] for a in df_dist_real_norm['tag1'].values]
        df_dist_real_norm['barrier2'] = [tags[int(a)] for a in df_dist_real_norm['tag2'].values]
        df_dist_real_norm['day'] = t_dist_real_norm[:,3]
        df_dist_real_norm['distance'] = t_dist_real_norm[:,0]
        df_dist_real_norm = df_dist_real_norm.sort_values(by='distance')
        df_dist_real['tag1']= t_dist_real[:,1].astype(int)
        df_dist_real['tag2']= t_dist_real[:,2].astype(int)#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_real['barrier1'] = [tags[int(a)] for a in df_dist_real['tag1'].values]
        df_dist_real['barrier2'] = [tags[int(a)] for a in df_dist_real['tag2'].values]
        df_dist_real['day'] = t_dist_real[:,3]
        df_dist_real['distance'] = t_dist_real[:,0]
        df_dist_real = df_dist_real.sort_values(by='distance')
        df_dist_real_norm = df_dist_real_norm.sort_values(by='distance')
        self.df_dist_real = df_dist_real
        self.df_dist_real_norm = df_dist_real_norm
        if sim.average_fluxes:
            df_dist_real.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg.csv'),';')
            df_dist_real_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg.csv'),';')
        else:
            df_dist_real.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg_{}.csv'.format(self.str0)),';')
            df_dist_real_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg_{}.csv'.format(self.str0)),';')
        #### SIMULATION ####
        df_dist_sim_norm = pd.DataFrame()
        df_dist_sim = pd.DataFrame()
        df_dist_sim_norm['tag1']= t_dist_sim_norm[:,1].astype(int)
        df_dist_sim_norm['tag2']= t_dist_sim_norm[:,2].astype(int)#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_sim_norm['barrier1'] = [tags[int(a)] for a in df_dist_real_norm['tag1'].values]
        df_dist_sim_norm['barrier2'] = [tags[int(a)] for a in df_dist_real_norm['tag2'].values]
        df_dist_sim_norm['day'] = t_dist_sim_norm[:,3]
        df_dist_sim_norm['distance'] = t_dist_sim_norm[:,0]
        df_dist_sim_norm = df_dist_sim_norm.sort_values(by='distance')
        df_dist_sim['tag1']= t_dist_sim[:,1].astype(int)
        df_dist_sim['tag2']= t_dist_sim[:,2].astype(int)#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
        df_dist_sim['barrier1'] = [tags[int(a)] for a in df_dist_real['tag1'].values]
        df_dist_sim['barrier2'] = [tags[int(a)] for a in df_dist_real['tag2'].values]
        df_dist_sim['day'] = t_dist_sim[:,3]
        df_dist_sim['distance'] = t_dist_sim[:,0]
        df_dist_sim = df_dist_sim.sort_values(by='distance')
        df_dist_sim_norm = df_dist_sim_norm.sort_values(by='distance')
        self.df_dist_sim = df_dist_sim
        self.df_dist_sim_norm = df_dist_sim_norm
        if sim.average_fluxes:
            df_dist_sim.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg_sim.csv'),';')
            df_dist_sim_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg_sim.csv'),';')
        else:
            df_dist_sim.to_csv(os.path.join(sim.state_basename,'euclidean_distance_avg_{}_sim.csv'.format(self.str0)),';')
            df_dist_real_norm.to_csv(os.path.join(sim.state_basename,'normed_flux_euclidean_distance_avg_{}_sim.csv'.format(self.str0)),';')
        self.dict_df_ward = {'df_dist_sim':df_dist_sim,'df_dist_sim_norm':df_dist_sim_norm,'df_dist_real':df_dist_real,'df_dist_real_norm':df_dist_real_norm}

    def correlation_matrix_plot(self,sim):
        '''Calculate the correlation matrices between different simulations'''
        corrMatrix = self.sim_dataframe.drop(['datetime','timestamp'],axis = 1).corr()
        corrMatrix1 = self.df_barriers.drop('timestamp',axis = 1).corr()
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(corrMatrix)
        plt.title('correlation matrix sim')
        plt.show()
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(corrMatrix1)
        plt.title('correlation matrix real')
        plt.show()
        self.dif_cor = corrMatrix - corrMatrix1
        fig = plt.figure(figsize=(30, 15))
        sns.heatmap(self.dif_cor)
        plt.title('difference in correlation matrix sim-real data')
        if not os.path.exists(os.path.join(sim.state_basename,'plots'))
            os.mkdir(os.path.join(sim.state_basename,'plots'))
        plt.savefig(os.path.join(sim.state_basename,'plots','difference_correlation_matrices_real_sim.png'))
        plt.show()
        self.cost_corr = 0.5*sum([sum(x) for x in self.dif_cor.fillna(0).to_numpy()])

    def ward_plot(self,sim):
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

        '''
        self.dist_type = 'euclidean'
        self.hca_method = 'ward'
        self.nan_t = 5 # %. droppa le barriers con troppi nan
        self.sm_window = 9 # smoothing window, deve essere dispari
        self.sm_order = 3 # smoothing polynomial order
        iconcolors = ['purple', 'orange', 'red', 'green', 'blue', 'pink', 'beige', 'darkred', 'darkpurple', 'lightblue', 'lightgreen', 'cadetblue', 'lightgray', 'gray', 'darkgreen', 'white', 'darkblue', 'lightred', 'black']
        self.dict_distance_ward = dict.fromkeys(list(self.dict_df_ward.keys()))
        self.dict_symmetric_distance_ward = dict.fromkeys(list(self.dict_df_ward.keys()))
        
        
        for k in list(self.dict_df_ward.keys()):
            fig = plt.figure(figsize=(30, 20))
            ax = fig.add_subplot(111)
            linkg = sch.linkage(self.dict_df_ward[k]['distance'].values,method=self.hca_method,optimal_ordering = True)
            self.dict_distance_ward[k], self.dict_symmetric_distance_ward[k] = calculate_distance_ward(linkg,sim.tags)
            sch.dendrogram(linkg, labels=sim.tags)
            plt.title(f'HCA dendrogram method: {self.hca_method}, case: {k}')
            plt.xticks(fontsize=7)
            plt.savefig(os.path.join(sim.state_basename,f'HCA dendrogram method_{hca_method}_case_{k}.png'))
            plt.show()
        self.symmetric_ward_distance_sim_real = 0
        self.symmetric_ward_distance_sim_real_norm = 0
        self.ward_distance_sim_real = 0
        self.ward_distance_sim_real = 0
        'df_dist_sim':df_dist_sim,'df_dist_sim_norm':df_dist_sim_norm,'df_dist_real':df_dist_real,'df_dist_real_norm'
        for k in list(self.dict_df_ward.keys()):
            for k1 in list(self.dict_df_ward.keys()):
                if k == 'df_dist_sim' and k2 == 'df_dist_real':
                    for key in list(self.dict_symmetric_distance_ward[k]):
                        self.symmetric_ward_distance_sim_real = self.symmetric_ward_distance_sim_real + self.dict_symmetric_distance_ward[k][key]
                        self.ward_distance_sim_real = self.ward_distance_sim_real  + self.dict_distance_ward[k][key]
                    self.symmetric_ward_distance_sim_real = self.symmetric_ward_distance_sim_real/2
                    self.ward_distance_sim_real = self.ward_distance_sim_real/2
                if k == 'df_dist_sim_norm' and k2 == 'df_dist_real_norm':
                    for key in list(self.dict_symmetric_distance_ward[k]):
                        self.symmetric_ward_distance_sim_real_norm = self.symmetric_ward_distance_sim_real_norm + self.dict_symmetric_distance_ward[k][key]
                        self.ward_distance_sim_real = self.ward_distance_sim_real + self.dict_symmetric_distance_ward[k][key]
                    self.symmetric_ward_distance_sim_real_norm = self.symmetric_ward_distance_sim_real_norm/2
                    self.ward_distance_sim_real = self.ward_distance_sim_real/2

    def calculate_distance_ward(z,labels):
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
        l = list(np.arange(labels))                                     
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

        print(dict_binary[0][1:-1])
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
                                dict_symmetric_distance[str(k0)+'-'+str(k1)] = max([c,d])
                                break
                            d = d + 1
                    c = c +1


        ward_distance = 0 
        for key in list(dict_symmetric_distance.keys()):
            ward_distance = ward_distance + dict_symmetric_distance[key]
        ward_distance = ward_distance/2

        return dict_symmetric_distance,dict_distance

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
