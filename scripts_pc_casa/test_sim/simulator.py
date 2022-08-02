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
    local_distribution: object -> np.normal() -> default none
    new_source_list: str -> ['Scalzi_2','Scalzi_3'] (list of new sources further COSTITUZIONE,PAPPADOPOLI,SALVATORE)
    new_source_bool: bool -> True
    reset_source_list: str -> 'Costituzione_IN-Papadopoli_IN-Schiavoni_IN'
    reset_source_bool: bool -> True
    change_source_list: str -> 'Papadopoli_IN'
    change_source_bool: bool -> True
    name_attractions: str -> 'Farsetti_1'
    add_attractions_bool: bool -> True
    '''
    def __init__(self,
    config = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','conf_venezia.json'),
    config0 = os.path.join(os.environ['WORKSPACE'],'slides','pvt','conf','conf.json.local.albi.make'),
    file_distances_real_data = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','COVE flussi_pedonali 18-27 luglio.xlsx'),
    dir_data = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','barriers_config.csv'),
    state_basename = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','output'),
    start_date = '2021-07-14 23:00:00',
    stop_date ='2021-07-21 23:00:00',
    average_fluxes = True ,
    attraction_activate = True,
    locals_ = True,
    local_distribution = 'none',
    new_source_list = ['Scalzi_2','Scalzi_3'],
    new_source_bool = True,
    reset_source_list = ['Costituzione_IN','Papadopoli_IN','Schiavoni_IN'],
    reset_source_bool = True,
    change_source_list = 'Papadopoli_IN',
    change_source_bool = True,
    name_attracctions = 'Farsetti_1' ,
    add_attractions_bool = True
    ):
    # FILE OF INTEREST
        with open(config, encoding='utf8') as f:
            self.simcfgorig = json.load(f)
        with open(config0, encoding='utf8') as g:
            self.simcfgorig0 = json.load(g)
        self.df=pd.read_excel(file_distances_real_data, engine='openpyxl')
        self.dir_data = dir_data
        self.data_barriers = pd.read_csv(self.dir_data,';')
    # VALIDATION SET 
        self.average_fluxes = average_fluxes
        self.attraction_activate = attraction_activate
                    ### SIMULATION SETTINGS ####
    # SAVING DICT        
        self.state_basename = state_basename
    # DATETIME SETTING
        self.time_format='%Y-%m-%d %H:%M:%S'
        self.start_date = start_date
        self.stop_date = stop_date
    # LOCALS 
        self.locals = locals_
        self.local_distribution = local_distribution
    # SOURCES  
        self.new_source_list = new_source_list
        self.new_source_bool = new_source_bool
        self.reset_source_list = reset_source_list
        self.reset_source_bool = reset_source_bool        
        self.change_source_list = change_source_list
        self.change_source_bool = change_source_bool
    # ATTRACTIONS
        self.name_attractions = name_attracctions
        self.add_attractions_bool = add_attractions_bool

            ##### HANDLING DF (EXCEL FILE FROM REGION) #######



    def get_sources(self):
        print(self.simcfgorig['sources'])
        return self
    def pick_day_input(self):
        group=self.df.groupby('varco')
        self.df_new=pd.DataFrame()
        for g in group.groups:
            df_temp=group.get_group(g)
            try:
                self.df_new['datetime']=np.array(df_temp['timestamp'].copy(),dtype='datetime64[s]')
                self.df_new[df_temp.iloc[0]['varco']+'_IN']=np.array(df_temp['direzione'].copy(),dtype=int)
                self.df_new[df_temp.iloc[0]['varco']+'_OUT']=np.array(df_temp['Unnamed: 3'].copy(),dtype=int)  
            except:
                pass
            del df_temp
            self.start_date = datetime.strptime(self.start_date,self.time_format)
            mask_day = [True if h.day==self.start_date.day else False for h in self.df_new.datetime]
            self.df_new = self.df_new.loc[mask_day]
            print('df new',self.df_new)
        return self

    def averaging_fluxes(self):
        group=self.df.groupby('varco')
        self.df_new=pd.DataFrame()
        for g in group.groups:
            df_temp=group.get_group(g)
            try:
                self.df_new['timestamp']=np.array(df_temp['timestamp'].copy(),dtype='datetime64[s]')
                self.df_new[df_temp.iloc[0]['varco']+'_IN']=np.array(df_temp['direzione'].copy(),dtype=int)
                self.df_new[df_temp.iloc[0]['varco']+'_OUT']=np.array(df_temp['Unnamed: 3'].copy(),dtype=int)  
            except:
                pass
        del df_temp
        self.df=self.df_new
        h_mask = np.arange(25)
        self.tags = list(self.df.columns.drop('timestamp'))
        self.tagn = len(self.tags)
        self.df_avg = pd.DataFrame(data=np.zeros([25,self.tagn+1]),columns=self.df.columns)
        week_mask = [True if h.weekday()<4 else False for h in self.df.timestamp]
        dfc_temp = self.df.loc[week_mask]
        for e in h_mask:
            mask = [True if h.hour==e else False for h in dfc_temp.timestamp]                                          
            self.df_avg.iloc[e][self.tags] = dfc_temp[self.tags].loc[mask].mean()
            self.df_avg.iloc[e]['timestamp'] = e
        return self 

    def set_simcfg_fluxes_from_list_sources(self):
        '''Mi sto riferendo nel caso d'uso a questa lista in input ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_1_IN']'''
        self.list_name_sources = list_name_sources
        if self.average_fluxes:
            self.df_avg = self.df_avg[:-1]
        else:
            pass
        list_fluxes_conf_file = []
        for name_source in self.list_name_sources:
            list_flux_single_source = []
            print('extract sources fluxes name source',name_source)
            for flux in self.df_avg[name_source]:
                for i in range(4):
                    list_flux_single_source.append(flux/4)
            if len(list_flux_single_source)!= 96:
                print('la lunghezza della lista per la sorgente {0} è {1}'.format(name_source,len(list_flux_single_source)))
                exit( )
            list_fluxes_conf_file.append(list_flux_single_source)
            if name_source in ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_1_IN']:
                name_source = name_source.split('_')[0]+'_'+ name_source.split('_')[2]
            else:
                pass
            self.new_source_name = ''
            for position_letter in range(len(name_source)):
                if position_letter == 0 or position_letter ==len(name_source)-1 or position_letter ==len(name_source)-2:
                    self.new_source_name = self.new_source_name + name_source[position_letter]
                else:
                    self.new_source_name = self.new_source_name + name_source[position_letter].lower() 
            self.simcfgorig['sources'][self.new_source_name]['creation_rate'] = list_flux_single_source
            print('extract sources fluxes simcfgorig',self.simcfgorig['sources'][self.new_source_name]['creation_rate'])
            return self


    def add_source(self):
        ''' adds directly the sources from list_flux_single_source that contains the fluxes of df_avg, the one with columns all the sources.'''
        for name_source in self.new_source_list:
            list_flux_single_source = []
            for flux in range(len(self.df_avg[name_source.upper() + '_IN'])):
                for i in range(4):
                    list_flux_single_source.append(list(self.df_avg[name_source.upper() + '_IN'])[flux]/4)
            if len(list_flux_single_source)!= 96:
                print('la lunghezza della lista per la sorgente {0} è {1}'.format(name_source,len(list_flux_single_source)))
                exit( )  
            self.simcfgorig['sources'][name_source + '_IN'] = {
            'creation_dt' : self.simcfgorig['sources']['Costituzione_IN']['creation_dt'],
            'creation_rate' : list_flux_single_source,
            'source_location' : {
            'lat' : self.data_barriers.loc[self.data_barriers['Description']==name_source.upper()+'_IN'].iloc[0]['Lat'],
            'lon' : self.data_barriers.loc[self.data_barriers['Description']==name_source.upper()+'_IN'].iloc[0]['Lon']
            },
            'pawns_from_weight': {
            'tourist' : {
                'beta_bp_miss' : self.simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['beta_bp_miss'],
                'speed_mps'    : self.simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['speed_mps']
            }
            }
        }
        return True
    def reset_source(self):
        self.reset_source = self.reset_source_list[2]
        list_zero=np.zeros(96).tolist()
        self.simcfgorig['sources'][self.reset_source]['creation_rate'] = list_zero
        print(self.simcfgorig['sources'])
        return True

    def sources_simulation_with_people(self):      
        intersect =set(self.reset_source_list).symmetric_difference(set(self.simcfgorig['sources'].keys()))
        print('intersect',intersect)
        intersect.discard('LOCALS')
        #self.list_sources = '-'.join(intersect)
        self.list_name_sources = list(intersect)
        print('new list of sources',self.list_name_sources)
        return self

    def create_list_delta_u_attraction(self):
        attraction_perturbation_activate= self.attraction_activate
        if attraction_perturbation_activate: 
            self.delta_u = [0.3, 0.25,-0.05,-0.1,-0.15, -0.2,-0.25,-0.3]
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

