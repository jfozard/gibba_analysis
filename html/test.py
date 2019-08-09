

"""
Following image import, find the projection surface of the bladder
and segment the cells
"""

import sys
import numpy as np
import scipy.ndimage as nd
import matplotlib as mpl
from matplotlib.colors import colorConverter
import matplotlib.pylab as plt
import seaborn as sns
import os.path

from PIL import Image
import pandas as pd

from bokeh.resources import CDN
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import HoverTool
from bokeh.models import OpenURL, TapTool, CustomJS, ColumnDataSource
from bokeh.embed import autoload_static
from bokeh.layouts import layout, gridplot
from bokeh.models.widgets import Select

from string import Template

import subprocess

template_3d = 'template.html'

def main():

    with open(template_3d, 'r') as g2:
        s = g2.read()
        t = Template(s)
        with open('test-3d.html', 'w') as f2:
            f2.write(t.substitute({'name':'test', 
                          'filename':'test.ply',
                          'csvname':'test.csv'}))

main()
