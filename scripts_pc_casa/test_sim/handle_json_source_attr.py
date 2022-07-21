import sys 
import os

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    import setting_functions_sim
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

def add_source(df_avg,simcfgorig,name_source):
  ''' adds directly the sources from list_flux_single_source that contains the fluxes of df_avg, the one with columns all the sources.'''
  data_barriers = pd.read_csv(dir_data,';')
  list_flux_single_source = []
  for flux in range(len(df_avg[name_source.upper() + '_IN'])):
    for i in range(4):
      list_flux_single_source.append(list(df_avg[name_source.upper() + '_IN'])[flux]/4)
  if len(list_flux_single_source)!= 96:
    print('la lunghezza della lista per la sorgente {0} è {1}'.format(name_source,len(list_flux_single_source)))
    exit( )  
  simcfgorig['sources'][name_source + '_IN'] = {
        'creation_dt' : simcfgorig['sources']['Costituzione_IN']['creation_dt'],
        'creation_rate' : list_flux_single_source,
        'source_location' : {
          'lat' : data_barriers.loc[data_barriers['Description']==name_source.upper()+'_IN'].iloc[0]['Lat'],
          'lon' : data_barriers.loc[data_barriers['Description']==name_source.upper()+'_IN'].iloc[0]['Lon']
        },
        'pawns_from_weight': {
          'tourist' : {
            'beta_bp_miss' : simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['beta_bp_miss'],
            'speed_mps'    : simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['speed_mps']
          }
        }
      }
  return True

def reset_source(simcfgorig,reset_source):
      list_zero=np.zeros(96).tolist()
      simcfgorig['sources'][reset_source]['creation_rate'] = list_zero
      return True

def sources_simulation_with_people(simcfgorig):      
      intersect =set(args.reset_source_list.split('-')).symmetric_difference(set(simcfgorig['sources'].keys()))
      intersect.discard('LOCALS')
      list_sources = '-'.join(intersect)
      print('new list of sources',list_sources)
      return list_sources

def dealing_sources(new_source_bool,name_source_list,df_avg,simcfgorig,locals_bool):    
  print('dealing sources')
  if locals_bool:
    no_local_array=np.zeros(96).tolist()
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array    
  if new_source_bool:
    name_source_list = args.new_source_list.split('-')
    for name_source in name_source_list: 
      print('add source',name_source)
      add_source(df_avg,simcfgorig,name_source)
#    print('il file di configurazione con sorgenti aggiunte è \n',simcfgorig['sources'])
  if reset_source_bool:
    reset_source_list = args.reset_source_list.split('-')
    print('resetto le sorgenti', reset_source_list)
    for r_source in reset_source_list: 
      reset_source(simcfgorig,r_source)
#    print('il file di configurazione è con sorgenti resettate \n',simcfgorig['sources'])

  f= False
  if f:
    temporary_sources= args.change_source_list.split('-')
    l_sources_to_set = []
    for s in temporary_sources:
      l_sources_to_set.append(s.upper())
    set_simcfg_fluxes_from_list_sources(df_avg,simcfgorig,l_sources_to_set) # default ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN'] #['SCALZI_3_IN','SCALZI_2_IN']
  return True    

def add_attraction(simcfgorig):
  data_barriers = pd.read_csv(dir_data,';')      
  if args.add_attractions_bool:
    name_attractions = args.name_attractions.split('-')        
    for name_attraction in name_attractions:
          simcfgorig['attractions'][name_attraction] = {
          'lat' : data_barriers.loc[data_barriers['Description']==name_attraction.upper()+'_IN'].iloc[0]['Lat'],
          'lon' : data_barriers.loc[data_barriers['Description']==name_attraction.upper()+'_IN'].iloc[0]['Lon'],
          'weight': list(np.ones(24)*0.6),
          'timecap':list(np.ones(24)*1000),
          'visit_time': 2880   
          }
  print(simcfgorig['attractions'][name_attraction])
  return True
