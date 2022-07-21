#! /usr/bin/env python3
import sys
import os
import json

try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

print(dir(simulation))

#simconf = {}
#s = simulation(json.dumps(simconf))
#print(s.sim_info())
#s.run()
