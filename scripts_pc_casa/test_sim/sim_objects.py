#### THESE CLASSES ARE USED FOR THE PREPROCESSING TO BE GIVEN TO THE SIMULATOR ####
try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    import simulator
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e


class configuration_handler:
    '''Configuration handler class:
    ---------------
    Input:
    list_sources: list -> ['Pappadopoli_IN','Costituzione_IN']
    list_attractions: list -> ['Farsetti_1']
    list_modified_attractions: list ->
    list_delta_u_attractions: list -> [['Farsetti_1',0.05],['Farsetti_1',-0.05],...] (list.shape() = (len(list_modified_attractions),1))
    list_added_sources: list -> ['Scalzi_1_IN,Scalzi_2_IN']
    list_resetted_sources: list -> ['Schiavoni_IN']
    config: str -> /path/to/conf_venezia.json
    config0: str -> /path/to/conf.json.local.albi.make
    simcfgorig: file_json -> conf_venezia.json
    '''
    def __init__(self,
                 list_sources = [],
                 list_attractions = []
                 list_modified_attractions = []
                 list_delta_u_attractions = []
                 list_added_sources = []
                 list_resetted_sources = []
                 config = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files','conf_venezia.json'),
                 config0 = os.path.join(os.environ['WORKSPACE'],'slides','pvt','conf','conf.json.local.albi.make'),
                 file_distances_real_data = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','COVE flussi_pedonali 18-27 luglio.xlsx'),
                 dir_data = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','barriers_config.csv'),
                 state_basename = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','output'),
                 simcfgorig = {}):
        self.list_sources = list_sources
        self.list_attractions = list_attractions
        self.list_modified_attractions = list_modified_attractions
        self.list_delta_u_attractions = list_delta_u_attractions
        self.list_added_sources = list_added_sources
        self.list_resetted_sources = list_resetted_sources
    # FILE OF INTEREST
        with open(config, encoding='utf8') as f:
            self.simcfgorig = json.load(f)
        with open(config0, encoding='utf8') as g:
            self.simcfgorig0 = json.load(g)
        self.df=pd.read_excel(file_distances_real_data, engine='openpyxl')
        self.dir_data = dir_data
        self.data_barriers = pd.read_csv(self.dir_data,';')
        self.is_assigned_new_sources = False
        self.is_assigned_default_sources = False
        self.is_resetted_sources = False
    
    
    
    
        def assign_sources_json(self,simcfgorig):
        '''Input:
        simcfgorig: file_json with all sim infos
        USAGE:
        Initializes list_sources with sources from json default
        DESCRIPTION:
        mandatory before reset_sources(), assign_sources_to_json()
        '''
        for s_name in simcfgorig['sources']:
            s = source()
            s.name = s_name
            s.is_default = True
            for value in simcfgorig['sources'][s_name]:
                if type(simcfgorig['sources'][s_name][value]) == int:
                    s.creation_dt = simcfgorig['sources'][s_name][value]
                elif type(simcfgorig['sources'][s_name][value]) == list:
                    s.creation_rate = self.simcfgorig['sources'][s_name][value]
                elif type(simcfgorig['sources'][s_name][value]) == dict:
                    if value == 'source_location':
                        s.source_location = simcfgorig['sources'][s_name][value]
                    else: 
                        s.pawns_from_weight = simcfgorig['sources'][s_name][value]
            self.list_sources.append(s)
        self.is_assigned_default_sources = True
        return True
                 
                 
    def assign_new_sources(self,
                          list_added_sources,
                          data_barriers,
                          df_avg,
                          ):
        '''Input:
        list_added_sources: from the simulator
        data_barriers = file_csv containing barriers already opened
        df_avg = file_csv inherited from simulations containing validation data.
                  '''
        for s_ in list_added_sources:
            s = source()
            s.name = s_
            s.is_added = True
            s.creation_dt = 30
            s.source_location = {'lat':data_barriers.loc[data_barriers['Description']==s_.upper()+'_IN'].iloc[0]['Lat'],'lon':data_barriers.loc[data_barriers['Description']==s_.upper()+'_IN'].iloc[0]['Lon']}
            s.pawns_from_weight = {
          'tourist':{'beta_bp_miss' : self.simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['beta_bp_miss'],
            'speed_mps': self.simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['speed_mps']}}
            for flux in range(len(df_avg[s_.upper() + '_IN'])):
                for i in range(4):
                    s.creation_rate = list(df_avg[s_.upper() + '_IN'])[flux]/4
                    if len(s.creation_rate)!= 96:
                        print('la lunghezza della lista per la sorgente {0} è {1}'.format(s_,len(s.creation_rate)))
                        exit( )
            list_sources.append(s)
        self.is_assigned_new_sources = True
        return True
    def reset_sources(self,reset_source_list,data_barriers):
        '''Input: 
        reset_source_list: from the simulator
        data_barriers = file_csv containing barriers already opened
        '''
        for s_ in reset_source_list:
            s = source()
            s.name = s_
            s.is_reset = True
            s.is_added = False
            s.is_default = False
            s.is_changed = False
            s.pawns_from_weight = {'tourist':{'beta_bp_miss' : self.simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['beta_bp_miss'],
            'speed_mps': self.simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['speed_mps']}}
            s.source_location = {'lat':data_barriers.loc[data_barriers['Description']==s_.upper()+'_IN'].iloc[0]['Lat'],'lon':data_barriers.loc[data_barriers['Description']==s_.upper()+'_IN'].iloc[0]['Lon']}
        self.list_sources.append(s)
        self.is_resetted_sources = True

    
     

    def assign_sources_to_simcfgorig(self,simcfgorig):
        '''Input: 
        simcfgorig: json_file
        Description: for each source in list_sources adds all the infos. Is called when all the '''
        for s in self.list_sources:
            if s.is_added == True or s.is_changed == True:
                simcfgorig['sources'][s.name + '_IN'] = {'creation_dt' : 30,'creation_rate' : s.creation_rate,
        'source_location' : {'lat' : s.source_location['lat'],'lon' : s.source_location['lon']},
        'pawns_from_weight': s.pawns_from_weight}
            elif s.is_reset == True:
                simcfgorig['sources'][s.name + '_IN'] = {'creation_dt' : 30,'creation_rate' : np.zeros(96).tolist(),
        'source_location' : {'lat' : s.source_location['lat'],'lon' : s.source_location['lon']},
        'pawns_from_weight': s.pawns_from_weight}
        return simcfgorig
                
                 
    def assign_attractions(self):
        ''' I assign the list_attractions from the configuration file'''
        for a_name in self.simcfgorig['attractions']:
            a = attraction()
            a.name = a_name
            for value in self.simcfgorig['attractions'][a_name]:
                if type(self.simcfgorig['attractions'][a_name][value]) == float:
                    if value == 'lat':
                        a.lat = self.simcfgorig['attractions'][a_name][value]
                    elif value == 'lon':
                        a.lon = self.simcfgorig['attractions'][a_name][value]
                    elif value == 'visit_time':
                        a.visit_time = self.simcfgorig['attractions'][a_name][value]
                else:
                    if value == 'weight':
                        a.weight = self.simcfgorig['attractions'][a_name][value]
                    else:
                        a.time_cap = self.simcfgorig['attractions'][a_name][value]
            self.list_attractions.append(a)
        return True

    def initialize_modification_lists(self,
                                      attraction_activate,
                                      locals_,
                                      local_distribution,
                                      new_source_list,
                                      new_source_bool,
                                      reset_source_list,
                                      reset_source_bool,
                                      change_source_list,
                                      change_source_bool,
                                      name_attracctions,
                                      add_attractions_bool):
                 
    
        



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
        self.list_attributes = [name, creation_dt, creation_rate,source_location, paswns_from_weight]
        self.name = name
        self.creation_dt = creation_dt
        self.creation_rate = creation_rate
        self.source_location = source_location
        self.pawns_from_weight = pawns_from_weight
        self.is_added = False
        self.is_default = False
        self.is_reset = False
        self.is_changed = False
    
    
    
    
    
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
                 name = ''
                 lat = 0.
                 lon = 0.
                 weight = []
                 time_cap = []
                 visit_time = 0.
                 
                ):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.weight = weight
        self.time_cap = time_cap
        self.visit_time = visit_time
        self.is_added =False
        self.is_changed = False
        