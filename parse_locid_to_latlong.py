import pandas as pd
import os
import pickle as cp
import numpy as np
data_path = os.path.join('..','datathon_data')

zones = pd.read_csv(os.path.join(data_path,'zones.csv'),index_col='location_id')
NTA_new = cp.load(open('NTA_new','rb'))

import collections
import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


@memoized
def location_id_to_latlong(id):
    NTA = zones.loc[id,'nta_code']
    coords = NTA_new.loc[NTA]
    return coords['LAT'],coords['LONG']