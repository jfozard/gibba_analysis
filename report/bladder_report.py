 

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

template_3d = 'html/template.html'
template_overlay = 'html/template_overlay.html'

def bladder_report(output_filename, name, input_ply, input_csv):

    title = name

    f = open(output_filename, 'w')
    f.write("""<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title> {} </title>
    </head>
    <body>
    """.format(title))

    df = pd.read_csv(input_csv)

    df = df.loc[df['label']>0]
    
    p = figure(plot_width=250, plot_height=250, x_axis_label='area', y_axis_label='aniso')
    line1 = p.circle('area', 'aniso', size=2, source=df)
    
    p1 = figure(plot_width=250, plot_height=250, x_axis_label='area')
    hist, edges = np.histogram(df['area'], density=True, bins=50)
    
    p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])

    p2 = figure(plot_width=250, plot_height=250, x_axis_label='aniso')
    hist, edges = np.histogram(df['aniso'], density=True, bins=50)
    p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])

    c = gridplot([[p, p1, p2]])

    js, tag = autoload_static(c, CDN, os.path.splitext(os.path.basename(output_filename))[0]+'-data.js')

    g = open(os.path.splitext(output_filename)[0]+'-data.js', 'w')
    g.write(js)
    

    f.write('<h2>'+title+'</h2>\n')
    f.write('<a href="'+name+'-3d-overlay.html"> 3D View </a>\n')
    f.write('<table>\n')
    f.write('<tr>\n')
    for heading in ['maxproj', 'labels', 'relarea', 'abslogarea', 'aniso']:
        f.write('<th>'+heading+'</th>\n')
    f.write('</tr>\n')
    f.write('<tr>\n')
    f.write('<td><a href="../maxproj/'+name+'.png"><img src="../maxproj/'+name+'-thumb.png" height="200px"></a></td>')
    f.write('<td><a href="../image_labels/'+name+'.png"><img src="../image_labels/'+name+'-thumb.png" height="200px"></a></td>')
    f.write('<td><a href="../image_relarea/'+name+'.png"><img src="../image_relarea/'+name+'-thumb.png" height="200px"></a></td>')
    f.write('<td><a href="../image_abslogarea/'+name+'.png"><img src="../image_abslogarea/'+name+'-thumb.png" height="200px"></a></td>')
    f.write('<td><a href="../image_aniso/'+name+'.png"><img src="../image_aniso/'+name+'-thumb.png" height="200px"></a></td>')
    f.write('</tr>\n')
    f.write('</table>\n')

    f.write(tag)


    df.to_html(f)
    f.write('</body>\n')


    with open(template_3d, 'r') as g2:
        s = g2.read()
        t = Template(s)
        with open(output_filename[:-5]+'-3d.html', 'w') as f2:
            f2.write(t.substitute({'name':name, 
                                   'filename':name+'.ply',
                                   'csvname':'../cell_data/'+name+'.csv'}))

    with open(template_overlay, 'r') as g2:
        s = g2.read()
        t = Template(s)
        with open(output_filename[:-5]+'-3d-overlay.html', 'w') as f3:
            f3.write(t.substitute({'name':name, 
                                   'filename':name+'.ply',
                                   'filename_extra':name+'_overlay.ply',
                                   'csvname':'../cell_data/'+name+'.csv',
                                   'default_aniso_min':0,
                                   'default_aniso_max':1,
                                   'default_area_min':25,
                                   'default_area_max':2500,
                                   }))


