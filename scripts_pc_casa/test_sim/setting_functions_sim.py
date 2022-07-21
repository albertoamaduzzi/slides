import sys
import os

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    from handle_json_source_attr import *
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

def pick_day_input(df,start_date):
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

#%%  
def average_fluxes(df):
#  ar_varc = np.unique(np.array(df['varco'],dtype=str))
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
  df=df_new
  h_mask = np.arange(25)
  tags = list(df.columns.drop('datetime'))
  tagn = len(tags)
  df_avg = pd.DataFrame(data=np.zeros([25,tagn+1]),columns=df.columns)
#  df_avg.set_index(keys=h_mask)
#  print('mask and index',df_avg.index)
  week_mask = [True if h.weekday()<4 else False for h in df.datetime]
  dfc_temp = df.loc[week_mask]
  for e in h_mask:
    mask = [True if h.hour==e else False for h in dfc_temp.datetime]                                          
    df_avg.iloc[e][tags] = dfc_temp[tags].loc[mask].mean()
    df_avg.iloc[e]['datetime'] = e
  return df_avg

def set_simcfg_fluxes_from_list_sources(df_avg,simcfgorig,list_name_sources):
  '''Mi sto riferendo nel caso d'uso a questa lista in input ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_1_IN']'''
  if args.averaging_fluxes:
    df_avg =df_avg[:-1]
  else:
    pass
  list_fluxes_conf_file = []
  for name_source in list_name_sources:
    list_flux_single_source = []
    print('extract sources fluxes name source',name_source)
    for flux in df_avg[name_source]:
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
    new_source_name = ''
    for position_letter in range(len(name_source)):
      if position_letter == 0 or position_letter ==len(name_source)-1 or position_letter ==len(name_source)-2:
        new_source_name = new_source_name + name_source[position_letter]
      else:
        new_source_name = new_source_name + name_source[position_letter].lower() 
    simcfgorig['sources'][new_source_name]['creation_rate'] = list_flux_single_source
    print('extract sources fluxes simcfgorig',simcfgorig['sources'][new_source_name]['creation_rate'])
    return simcfgorig
  


    
def create_list_delta_u_attraction(args):
  attraction_perturbation_activate=args.attraction_activate
  locals_bool = args.locals
  if attraction_perturbation_activate: 
    delta_u = [0.3, 0.25,-0.05,-0.1,-0.15, -0.2,-0.25,-0.3]
    attractions = list(simcfgorig['attractions'].keys())# I am returned with the list of attractions
  else:
    delta_u = [0]
    attractions = list(simcfgorig['attractions'].keys())# I am returned with the list of attractions

  # modifico i parametri di interesse (attractivity) 
  list_delta_u_attraction=[]
  for attraction in attractions:
      for w in delta_u:
            list_delta_u_attraction.append([attraction,w])
  return list_delta_u_attraction

def setting_state_base_name(att_act,locals_bool,new_source_bool,list_delta_u_attraction,simcfgorig,layer_directory_to_add):
# Initialising the change of the attractions
  if att_act:
    simcfgorig['attractions'][attraction]['weight'] = list(np.array(simcfgorig['attractions'][attraction]['weight']) + np.ones(len(np.array(simcfgorig['attractions'][attraction]['weight'])))*w)
#HANDLING CASES
  if locals_bool and att_act:
#
    if  list_delta_u_attraction[1]>0:
      saving_file = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1]).split('.')[1]+'_no_locals'         
    else:
          saving_file = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]+'_no_locals'
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
#
  elif locals_bool and not att_act:
    print('I am in the case no locals and no variation of attraction')
    no_local_array=np.zeros(96).tolist()
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
    saving_file = args.citytag + '_no_locals' #str(list_delta_u_attraction[1]) is simply 0         
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
    print('state_base_name:\t',simcfgorig['state_basename'])
#
  elif not locals_bool and att_act:
    no_local_array=np.zeros(96).tolist()  
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
    if  list_delta_u_attraction[1]>0:
      saving_file = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1])          
    else:
          saving_file = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
#
  else:
    saving_file = args.citytag         
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
  return True    


#Define function to be runned in parallel
def run_simulation(list_delta_u_attraction):
# initialize simcfgorig
  with open(os.path.join(conf_dir,'conf_venezia.json')) as g:
    simcfgorig = json.load(g)
# Answering: new sources? I want locals? Do I change the weight of the attractions 
  new_source_bool = args.new_source_bool        
  if new_source_bool:
    name_source_list = args.new_source_list
  locals_bool = args.locals
  att_act =args.attraction_activate
# Define the sources
  dealing_sources(new_source_bool,name_source_list,df_avg,simcfgorig,locals_bool)
  layer_directory_to_add = sources_simulation_with_people(simcfgorig)
# setting saving dir
  add_attraction(simcfgorig)
  setting_state_base_name(att_act,locals_bool,new_source_bool,list_delta_u_attraction,simcfgorig,layer_directory_to_add)
# lancio la simulazione
  cfg = conf(simcfgorig0)
  simcfg = cfg.generate(start_date = args.start_date, stop_date = args.stop_date, citytag = args.citytag)#(override_weights=w)
# Adding piece where I change the sources instead of conf.py
  simcfgs = json.dumps(simcfgorig)
  s = simulation(simcfgs)
  print(s.sim_info())
  s.run()
  return True  
