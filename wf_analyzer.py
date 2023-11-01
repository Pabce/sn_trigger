import numpy as np
import matplotlib.pyplot as plt
import uproot

dir = '/eos/project-e/ep-nu/pbarhama/sn_saves/prod_snnue_pds'
file = 'zhist.root'

file = uproot.open(dir + '/' + file)["opdigiana"]