
import sys
import csv
import os
import time
import subprocess
import tempfile

from report.summary_table_combine import summary_table
from report.summary_graph_bokeh import summary_graph_overall
from report.bladder_report import bladder_report
from report.maxproj import maxproj
import pandas as pd

import uuid

local_base_dir = '/home/fozardj/'
local_cwd = local_base_dir+'bladder_pipeline3/'
base_dir = '/run/user/1000/gvfs/smb-share:server=jic-hpc-data,share=hpc-home/' 
slurm_base_dir = '/hpc-home/fozardj/'
signal_dir = base_dir + 'bladder_new/signal/'


base_data_dir_2d = base_dir + 'bladder_new/data/'
base_data_dir_3d = base_dir + 'bladder_new/data3d/'
base_data_dir = base_dir + 'bladder_new/'

base_container_dir = slurm_base_dir +'bladder_new/containers/'

base_script_dir = 'scripts/'

temp_script_dir = base_dir + 'bladder_new/scripts/temp/'

clip_script_dir = base_script_dir + 'clip/'

slurm_run = local_cwd + base_script_dir + 'slurm_run_signal4.sh'

surfacespm_local = local_base_dir +'surface_spm/src/multi_tool.py'
headless_local = local_base_dir +'surface_spm/src/headless_tool.py'

analysecells_local = local_base_dir +'surface_spm/src/analyse_cells.py'
analyseall_local = local_base_dir +'surface_spm/src/analyse.py'

import os, errno

slurm_script = """sbatch -p rg-sv  <<EOF 
#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1             # How many tasks on each node
#SBATCH --time=100:00:00
#SBATCH --mem 60000
sleep 20
{}
touch {}
EOF
"""

slurm_script4 = """sbatch -p rg-sv  <<EOF 
#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1             # How many tasks on each node
#SBATCH --time=100:00:00
#SBATCH --mem 15000
sleep 20
{}
touch {}
EOF
"""


def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def modify_script(script, strings):
    if not os.path.exists(temp_script_dir):
        os.makedirs(temp_script_dir)

    fileTemp = tempfile.NamedTemporaryFile(delete = False, dir=temp_script_dir)
    f = open(script, 'r')
    s = f.read()
    f.close()
    fileTemp.write(s%tuple(strings))
    fileTemp.close()
    return fileTemp.name


class Record(object):
    def __init__(self):
        self.colnames = []
        self.data = []

    def load_csv(self, filename):
        print 'load', filename
        if ':' in filename:
            filename, r0, r1 = filename.split(':')
        else:
            r0 = r1 = None
        with open(filename, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            self.colnames = list(reader.fieldnames)
            self.data = []
            for line in reader:
                self.data.append(line)
        if r0 is not None:
            self.data = self.data[int(r0):int(r1)]
            
            
    def write_csv(self, filename):
        with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.colnames)
                writer.writeheader()
                for d in self.data:
                    writer.writerow(d)

    def print_data(self):
        print self.colnames
        for d in self.data:
            print ', '.join([d.get(c, '') for c in self.colnames]) 


    def add_process_column(self, process):
        if process.name not in self.colnames:
            self.colnames.append(process.name)


class AnalyseAll(object):
    def __init__(self):
        self.name = 'AnalyseAll'
        self.filename_colname = 'filename'
        self.params_colnames = ['DAI','HAI', 'class', 'trap_length']
        self.input_dirs = ['hc_clean2/', 'clean2/']
        self.output_dirs = []

    def run_process(self, records):

        names_3d = [os.path.splitext(d.get(self.filename_colname))[0] for d in records[0].data]
        names_2d = [os.path.splitext(d.get(self.filename_colname))[0] for d in records[1].data]

        
        seg_filenames_3d = [ base_data_dir_3d + self.input_dirs[0] + n + '.ply' for n in names_3d ]
        seg_filenames_2d = [ base_data_dir_2d + self.input_dirs[1] + n + '.ply.gz' for n in names_2d ]

        output_filename = base_data_dir + 'report.csv'
        
        gland_criterion = '"(norm_area<0.25) or (norm_area>4)"'

#        ensure_path_exists(base_data_dir + 'report')

        seg_filenames = seg_filenames_3d + seg_filenames_2d
        
#        print subprocess.check_output(['python2', analyseall_local, '-o',output_filename]+seg_filenames)

        extra_filename = base_data_dir + 'extended_report.csv'
        
# Prepend name, extra columns to CSV file

        df = pd.read_csv(output_filename)

        extra_3d = [[n]+map(d.get, self.params_colnames) for n, d in zip(names_3d, records[0].data)]
        extra_2d = [[n]+map(d.get, self.params_colnames) for n, d in zip(names_2d, records[1].data)]

        extra = extra_3d + extra_2d
        
        df2 = pd.DataFrame.from_records(extra, columns = ['name']+self.params_colnames)

        print df2.columns

        df3 = pd.concat([df2, df], axis=1) 

        df3.to_csv(extra_filename, index=False)
        
class SummaryTable(object):
    def __init__(self):
        self.name = 'SummaryTable'
        self.filename_colname = 'filename'
        self.params_colnames = ['DAI','HAI','class', 'trap_length']
        self.input_dirs = [] #'clean2/']

    def run_process(self, records):
        data = records[0].data + records[1].data
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in data]
        DAI = [map(d.get, self.params_colnames)[0] for d in data]
        HAI = [map(d.get, self.params_colnames)[1] for d in data]
        classes = [map(d.get, self.params_colnames)[2] for d in data]
        lengths = [map(d.get, self.params_colnames)[3] for d in data]
        output_filename = base_data_dir + 'index.html'
        input_filenames = [ 'data3d/hc_' for d in records[0].data ] +  [ 'data/' for d in records[1].data ]
#        print output_filename, input_filenames

        summary_table(output_filename, base_data_dir, input_filenames, DAI, HAI, classes, lengths, names)



class SummaryGraph(object):
    def __init__(self):
        self.name = 'SummaryGraph'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['']

    def run_process(self, records):
        output_filename = base_data_dir + 'graph.html'
        input_filename = base_data_dir + 'extended_report.csv'

        input_filenames = [ 'data3d/hc_' for d in records[0].data ] +  [ 'data/' for d in records[1].data ]

        summary_graph_overall(input_filename, output_filename, input_filenames)



def main():
    process_list = [AnalyseAll(), SummaryTable(), SummaryGraph()]
    process_dict = dict((p.name,p) for p in process_list)


    if len(sys.argv)>3:
        pipeline = sys.argv[3].split(',')
        if '+' in pipeline[0]:
            all_names = [process.name for process in process_list]
            pipeline = all_names[all_names.index(pipeline[0][:-1]):]
    else:
        pipeline = [process.name for process in process_list]


    r3 = Record()
    r3.load_csv(sys.argv[1])

    r = Record()
    r.load_csv(sys.argv[2])

    
    for pn in pipeline:
        p = process_dict[pn]
        print "Run Process", pn
        p.run_process((r3, r))

    

main()
