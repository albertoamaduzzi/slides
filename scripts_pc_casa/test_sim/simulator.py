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
    new_source_list: str -> 'Scalzi_2-Scalzi_3' (list of new sources further COSTITUZIONE,PAPPADOPOLI,SALVATORE)
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
    state_basename = os.path.join(os.environ['WORKSPACE'],'slides','work_ws','output'),
    start_date = '2021-07-14 23:00:00',
    stop_date ='2021-07-21 23:00:00',
    average_fluxes = True ,
    attraction_activate = True,
    locals = True,
    local_distribution = 'none',
    new_source_list = 'Scalzi_2-Scalzi_3',
    new_source_bool = True,
    reset_source_list = 'Costituzione_IN-Papadopoli_IN-Schiavoni_IN',
    reset_source_bool = True,
    change_source_list = 'Papadopoli_IN',
    change_source_bool = True,
    name_attracctions = 'Farsetti_1' ,
    add_attractions_bool = True
    ):
    # FILE OF INTEREST
        with open(config) as f:
            self.simcfgorig = json.load(f)
        with open(config0) as g:
            self.simcfgorig0 = json.load(g)
        self.real_data=pd.read_excel(file_distances_real_data, engine='openpyxl')
        self.state_basename = state_basename
        self.start_date = start_date
        self.stop_date = stop_date
        self.average_fluxes = average_fluxes
        self.attraction_activate = attraction_activate
        self.locals = locals
        self.local_distribution = local_distribution
        self.new_source_list = new_source_list
        self.new_source_bool = new_source_bool
        self.reset_source_list = reset_source_list
        self.reset_source_bool = reset_source_bool        
        self.change_source_list = change_source_list
        self.change_source_bool = change_source_bool
        self.name_attracctions = name_attracctions
        self.add_attractions_bool = add_attractions_bool

def 

def pick_day_input(self,df,start_date):
  '''df: is the file excel whose columns are: timestamp, varco, in,out.
  returns a dataframe whose columns are the sources and rows date-timed fluxes of the day that starts at start_date'''
  time_format='%Y-%m-%d %H:%M:%S'
#  ar_varc=np.unique(np.array(df['varco'],dtype=str))
  group=df.groupby('varco')
  df_new=pd.DataFrame()
  for g in group.groups:
    df_temp=group.get_group(g)
    try:
      df_new['datetime']=np.array(df_temp['timestamp'].copy(),dtype='datetime64[s]')
      df_new[df_temp.iloc[0]['varco']+'_IN']=np.array(df_temp['direzione'].copy(),dtype=int)
      df_new[df_temp.iloc[0]['varco']+'_OUT']=np.array(df_temp['Unnamed: 3'].copy(),dtype=int)  
    except:
      pass
  del df_temp
  start_date = datetime.strptime(start_date,time_format)
  mask_day = [True if h.day==start_date.day else False for h in df_new.datetime]
  df_new = df_new.loc[mask_day]
  print('df new',df_new)
  return df_new



