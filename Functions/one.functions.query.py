"""
Functions to query mice/sessions/trials using ONE-exclusive functions
Jan 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np
import datetime
import pickle 
import os
# %%
from one.api import ONE
#one = ONE(base_url='https://openalyx.internationalbrainlab.org')
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# %%
# TODO: write functions to query mice based on training criterion; make sure same subject are 
# queried on the DJ and ONE databases