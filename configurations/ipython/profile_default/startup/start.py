# The basic idea is to NOT have any regular package imports here.
# This just confuses students.
import numpy as np
from IPython import get_ipython
from IPython.core.display import HTML

ipython = get_ipython()

ipython.magic("load_ext autoreload")
ipython.magic("load_ext lab_black")

ipython.magic("matplotlib inline")
ipython.magic("autoreload 2")

np.set_printoptions(precision=4)
