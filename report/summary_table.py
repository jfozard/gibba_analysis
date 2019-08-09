

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

from maxproj import maxproj

from PIL import Image
import pandas as pd

import subprocess

def summary_table(output_filename, output_dir, input_filenames, DAI, HAI, size_classes, lengths, names):

    title = 'Bladder Segmentation'

    f = open(output_filename, 'w')
    f.write("""<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title> {} </title>
      <style>
      td {{ border: none }}
      </style>
    </head>
    <body>
    """.format(title))

    if title:
        f.write("<h2> {} </h2>\n".format(title))

    f.write('<h3> <a href="graph.html"> Graph of data </a> </h3>\n')

    bg_color = {'A':['#e0a0a0', '#d09090'],
                'B':['#e0a0a0', '#d09090'],
                'C':['#a0e0a0', '#90d090'],
                'D':['#a0e0a0', '#90d090'],
                'E':['#a0e0a0', '#90d090'],
                'F':['#a0a0e0', '#9090d0'],
                'G':['#a0a0e0', '#9090d0'],
                'H':['#a0a0e0', '#9090d0'],
                '':['#c0c0c0', '#c0c0c0']}

    f.write('<table style="width:100%">\n')
    f.write('<tr>\n')
    for heading in ['name', 'DAI', 'HAI', 'length', 'maxproj', 'labels', 'relarea', 'abslogarea', 'aniso']:
        f.write('<th>'+heading+'</th>\n')
    f.write('</tr>\n')
    
    for i, (fn,dai0, hai0, c,length, n) in enumerate(zip(input_filenames, DAI, HAI, size_classes, lengths, names)):

        c = 'A'
        dai0 = float(dai0)
        hai0 = float(hai0)
        length = float(length)
        
#        maxproj(output_dir+'reduce/'+n+'.tif', output_dir+'maxproj/'+n+'.png')

        """
        subprocess.check_output(['convert',output_dir+'maxproj/'+n+'.png', '-resize', '200', output_dir+'maxproj/'+n+'-thumb.png'])
        subprocess.check_output(['convert',output_dir+'image_labels/'+n+'.png', '-resize', '10%', output_dir+'image_labels/'+n+'-thumb.png'])
        subprocess.check_output(['convert',output_dir+'image_relarea/'+n+'.png', '-resize', '10%', output_dir+'image_relarea/'+n+'-thumb.png'])
        subprocess.check_output(['convert',output_dir+'image_abslogarea/'+n+'.png', '-resize', '10%', output_dir+'image_abslogarea/'+n+'-thumb.png'])
        subprocess.check_output(['convert',output_dir+'image_aniso/'+n+'.png', '-resize', '10%', output_dir+'image_aniso/'+n+'-thumb.png'])
        subprocess.check_output(['convert',output_dir+'image_aniso_chin/'+n+'.png', '-resize', '10%', output_dir+'image_aniso_chin/'+n+'-thumb.png'])
        """
        link_fn = 'bladder_report/'+n+'.html'
        f.write('<tr style="background-color:'+bg_color[c][i%2]+'"> <td> <a href="'+link_fn+'"> '+n[:20]+'</a></td><td>'+'{:.2f}'.format(dai0)+'</td><td>'+'{:.1f}'.format(hai0)+'</td><td>' +'{:.0f}'.format(length))#+'</td><td>'+c+'</td>')
        f.write('<td><a href="maxproj/'+n+'.png"><img src="maxproj/'+n+'-thumb.png" height="200px"></a></td>')
        f.write('<td><a href="image_labels/'+n+'.png"><img src="image_labels/'+n+'-thumb.png" height="200px"></a></td>')
        f.write('<td><a href="image_relarea/'+n+'.png"><img src="image_relarea/'+n+'-thumb.png" height="200px"></a></td>')
        f.write('<td><a href="image_abslogarea/'+n+'.png"><img src="image_abslogarea/'+n+'-thumb.png" height="200px"></a></td>')
        f.write('<td><a href="image_aniso/'+n+'.png"><img src="image_aniso/'+n+'-thumb.png" height="200px"></a></td>')
        #f.write('<td><a href="image_aniso_chin/'+n+'.png"><img src="image_aniso_chin/'+n+'-thumb.png" height="200px"></a></td>')
        f.write('</tr>\n')

    f.write('</table>\n')
    f.write('</body>\n')

