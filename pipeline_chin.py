
import sys
import csv
import os
import os.path
import time
import subprocess
import tempfile

from report.summary_table import summary_table
from report.summary_graph_bokeh import summary_graph
from report.bladder_report import bladder_report
from report.maxproj import maxproj
import pandas as pd

import uuid

local_base_dir = '/home/fozardj/'
local_cwd = local_base_dir+'bladder_pipeline3/'
base_dir = '/run/user/1000/gvfs/smb-share:server=jic-hpc-data,share=hpc-home/' 
slurm_base_dir = '/hpc-home/fozardj/'
signal_dir = base_dir + 'bladder_new/signal/'


base_data_dir = base_dir + 'bladder_new/data_chin/'
base_container_dir = slurm_base_dir +'bladder_new/containers/'

base_script_dir = 'scripts/'

temp_script_dir = base_dir + 'bladder_new/scripts/temp/'

clip_script_dir = base_script_dir + 'clip_chin/'

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

class Process(object):
    def __init__(self):
        self.name = ''
        self.filename_colname = None
        self.params_colnames = []
        self.input_dirs = []
        self.output_dirs = []
        
    def run(self, filename, params):

        filename = os.path.splitext(filename)[0]
        input_files = [base_data_dir+d+filename for d in self.input_dirs]
        output_files = [base_data_dir+d+filename for d in self.output_dirs]

        for directory in [base_data_dir+d for d in self.output_dirs]:
            if not os.path.exists(directory):
                os.makedirs(directory)



                
        process_id = str(uuid.uuid1())
        print 'exec ' + process_id

        if not os.path.exists(signal_dir):
            os.makedirs(signal_dir)
        
        sn = self.exec_process(input_files, output_files, params, signal_dir+process_id)
        if sn is not False:
            if sn:
                self.pending += sn
            else:
                self.pending.append(signal_dir+process_id)

        print "PENDING", self.pending


    def run_process(self, record):
        process = self
        self.pending = []
        for d in record.data:
            if d.get(process.name)!='done':
                filename = d[process.filename_colname]
                params = dict((c, d.get(c, None)) for c in process.params_colnames)
                process.run(filename, params)
                time.sleep(5)
                d[process.name] = time.asctime()
        self.wait_until_complete()

    def wait_until_complete(self):
        while True:
            print self.name
            print 'pending jobs: ', self.pending
            print 'NJ', len(self.pending)
            done = []
            for n in self.pending:
                if os.path.exists(n):
                    done.append(n)
            self.pending = [p for p in self.pending if p not in done]
            if not self.pending:
                return
            time.sleep(20)

def exec_slurm(commands, signal_name, size=5, base_script=slurm_script):
    
    p = subprocess.Popen(["ssh", "slurm"], stdin=subprocess.PIPE)
    signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
    all_script = ''
    all_signals = []
    for i in range(0, len(commands), size):
        new_commands = '\n'.join([ 'srun ' + ' '.join(c) for c in commands[i:i+size] ])
        script = base_script.format(new_commands, signal_name_slurm+str(i))
        all_signals.append(signal_name+str(i))
        all_script += script
        all_script += 'sleep 2\n'
    p.communicate(all_script)
    return all_signals



    
class CombineProcess(object):
    def __init__(self):
        self.name = ''
        self.filename_colname = None
        self.params_colnames = []
        self.input_dirs = []
        self.output_dirs = []
        self.base_script = slurm_script
        
    def run(self, filename, params):

        filename = os.path.splitext(filename)[0]
        input_files = [base_data_dir+d+filename for d in self.input_dirs]
        output_files = [base_data_dir+d+filename for d in self.output_dirs]

        return self.get_slurm_command(input_files, output_files, params)   
        


    def run_process(self, record):
        process = self
        self.pending = []

        for directory in [base_data_dir+d for d in self.output_dirs]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        process_id = str(uuid.uuid1())
        print 'exec ' + process_id

        if not os.path.exists(signal_dir):
            os.makedirs(signal_dir)

        slurm_commands = []
        for d in record.data:
            if d.get(process.name)!='done':
                filename = d[process.filename_colname]
                params = dict((c, d.get(c, None)) for c in process.params_colnames)
                slurm_commands.append(self.run(filename, params))
                d[process.name] = time.asctime()

#        self.pending.append(signal_dir + process_id)

        if hasattr(self, 'batch'):
            signals = exec_slurm(slurm_commands, signal_dir + process_id, size=self.batch)
        else:
            signals = exec_slurm(slurm_commands, signal_dir + process_id)
        self.pending += signals
        self.wait_until_complete()
                           
    def wait_until_complete(self):
        while True:
            print self.name
            print 'pending jobs: ', self.pending
            print 'NJ', len(self.pending)
            done = []
            for n in self.pending:
                if os.path.exists(n):
                    done.append(n)
            self.pending = [p for p in self.pending if p not in done]
            if not self.pending:
                return
            time.sleep(20)



class Enhance(Process):
    def __init__(self):
        self.name = 'Enhance'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['reduce/']
        self.output_dirs = ['enhance/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
        
            input_files_slurm = [ slurm_base_dir + f[len(base_dir):]+'.tif' for f in input_files]
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.tif'
        
            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]

            print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'pythonspm.img', '/python-spm/src/enhance/edge.py', input_files_slurm[0], output_file_slurm, ])

            return True
        else:
            print 'input not found', input_files
            return False
#        os.remove(edited_script)


class Project(Process):
    def __init__(self):
        self.name = 'Project'
        self.filename_colname = 'filename'
        self.params_colnames = ['Project_level']
        self.input_dirs = ['reduce/','enhance/']
        self.output_dirs = ['surface/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
            script_fn = base_script_dir + 'project_script.txt'

            print params['Project_level']
        
#            print edited_script

        
            input_files_slurm = [ slurm_base_dir + f[len(base_dir):]+'.tif' for f in input_files]
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply'

            edited_script = modify_script(script_fn, input_files_slurm + [params['Project_level']]+ [output_file_slurm])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

        
            print input_files_slurm, output_files

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]

            print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'meshproject.img', '-r', edited_script])


        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)
        

            
    

class MultiSegment(Process):
    def __init__(self):
        self.name = 'MultiSegment'
        self.filename_colname = 'filename'
        self.params_colnames = ['Segment_nbhoodsize', 'Segment_mean', 'Segment_grad']
        self.input_dirs = ['surface/']
        self.output_dirs = ['multisegment/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        all_signals = []
        if os.path.exists(input_files[0]+'.ply'):
        
            for seed in range(5):
                script_fn = base_script_dir + 'multisegment_script_chin.txt'
            
                print  params['Segment_nbhoodsize'], params['Segment_mean'], params['Segment_grad']
                

                edited_script = modify_script(script_fn, [seed, seed, seed, params['Segment_mean'], params['Segment_nbhoodsize'], params['Segment_grad']])
        
                print edited_script

                edited_script = slurm_base_dir + edited_script[len(base_dir):]
                
                input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply'
                output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.'+str(seed)+'.ply.gz'
        

                signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]+str(seed)
                all_signals.append(signal_name+str(seed))
                print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script, '-o', output_file_slurm, input_file_slurm])


        else:
            print 'input not found', input_files[0]
        return all_signals
#        os.remove(edited_script)


class MultiCombine(Process):
    def __init__(self):
        self.name = 'MultiCombine'
        self.filename_colname = 'filename'
        self.params_colnames = ['Combine_level']
        self.input_dirs = ['multisegment/']
        self.output_dirs = ['combine/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply.gz'

        if_list = [input_files[0]+'.'+str(i)+'.ply.gz' for i in range(5)]
        if all(os.path.exists(f) for f in if_list):
        
            input_files_slurm = [slurm_base_dir + f[len(base_dir):] for f in if_list]
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply.gz'
        
            print input_files_slurm

            print 'Combine level', params['Combine_level']

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]

            print subprocess.check_output([slurm_run, signal_name_slurm, 'singularity', 'run', '--app', 'combine', base_container_dir+'segment.img', params['Combine_level'], output_file_slurm] + input_files_slurm)

        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)




class Clean(Process):
    def __init__(self):
        self.name = 'Clean'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['combine/']
        self.output_dirs = ['clean/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
            script_fn = base_script_dir + 'clean_script.txt'
            

            edited_script = modify_script(script_fn, [])
        
            print edited_script

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply.gz'

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
        
            print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script, '-o', output_file_slurm, input_file_slurm])



        else:
            print 'input not found', input_files[0]


class Clip(Process):
    def __init__(self):
        self.name = 'Clip'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['clean/']
        self.output_dirs = ['clipped/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):

            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply.gz'


            script_fn = clip_script_dir + 'clip-'+os.path.basename(input_files[0])+'.txt'
            
            edited_script = modify_script(script_fn,[ input_file_slurm, output_file_slurm ])

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            print edited_script

        
            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]        

            print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script ])


        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)

class Clean2(Process):
    def __init__(self):
        self.name = 'Clean2'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['clipped/']
        self.output_dirs = ['clean2/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
            script_fn = base_script_dir + 'clean2.txt'
            

            edited_script = modify_script(script_fn, [])
        
            print edited_script

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply.gz'
        
            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]        

            print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script, '-o', output_file_slurm, input_file_slurm])



        else:
            print 'input not found', input_files[0]


class ImageProcess(CombineProcess):

    def __init__(self):
        self.script_fn = base_script_dir + 'borders.txt'
        self.extra_options = []
        
    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
        

            script_fn = self.script_fn #

            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]


            input_file_slurm = '"'+slurm_base_dir + input_files[0][len(base_dir):]+'.ply'+'"'
        


            all_commands = []
            for i, m in enumerate(views):
                if i>0:
                    output_file_slurm = '"'+slurm_base_dir + output_files[0][len(base_dir):]+'-'+str(i)+'.png'+'"'
                else:
                    output_file_slurm = '"'+slurm_base_dir + output_files[0][len(base_dir):]+'.png'+'"'
            
                all_commands += ['singularity', 'run', '--app', 'image',
                                 base_container_dir+'segment2.img', '--ip_scalebar', '100', '--i_matrix', m] \
                + self.extra_options + ['-s', edited_script, '-i', output_file_slurm, input_file_slurm + '\n']
            return all_commands

            
class ImageLabels(CombineProcess):
    def __init__(self):
        self.name = 'ImageLabels'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_labels/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):

            script_fn = base_script_dir + 'borders.txt'

            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]


            input_file = input_files[0]+'.ply.gz'
            output_file = output_files[0]+'.png'
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.png'
        

            return [ 'singularity', 'run', '--app', 'image',
                                           base_container_dir+'segment2.img', '--ip_scalebar', '100',
                                           '-s', edited_script, '-i', output_file_slurm, input_file_slurm]
            

           # print subprocess.check_output(['python', surfacespm_local, '-s', script_fn, '--ip_scalebar', '100', '-i',output_file, input_file])



        else:
            print 'input not found', input_files[0]
#        open(signal_name, 'a').close()
#        os.remove(edited_script)


class ImageRelArea(CombineProcess):
    def __init__(self):
        self.name = 'ImageRelArea'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_relarea/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
        

            script_fn = base_script_dir + 'borders.txt'

            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            input_file = input_files[0]+'.ply.gz'
            output_file = output_files[0]+'.png'
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.png'
        

            return ['singularity', 'run', '--app', 'image',
                                           base_container_dir+'segment2.img', '--ip_scalebar', '100', '--ip', 'area', '--ip_colmap', 'new_jet',
                                           '-s', edited_script, '-i', output_file_slurm, input_file_slurm]
            


        else:
            print 'input not found', input_files[0]

class ImageAbsLogArea(CombineProcess):
    def __init__(self):
        self.name = 'ImageAbsLogArea'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_abslogarea/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
        

            script_fn = base_script_dir + 'borders.txt'

            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]


            input_file = input_files[0]+'.ply.gz'
            output_file = output_files[0]+'.png'
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.png'
        

            return ['singularity', 'run', '--app', 'image',
                                           base_container_dir+'segment2.img', '--ip_scalebar', '100', '--ip', 'area', '--ip_colmap', 'new_jet', '--ip_range', '25,2500', '--ip_log',
                                           '-s', edited_script, '-i', output_file_slurm, input_file_slurm]
            


#            print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--ip_scalebar', '100', '-i',output_file, input_file])

        else:
            print 'input not found', input_files[0]


class ImageAniso(CombineProcess):
    def __init__(self):
        self.name = 'ImageAniso'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_aniso/', 'image_aniso_chin/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
        

            script_fn = base_script_dir + 'aniso_vectors.txt'

            
            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]


            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.png'
            output_file_chin_slurm = slurm_base_dir + output_files[1][len(base_dir):]+'.png'
        

            return ['singularity', 'run', '--app', 'image',
                                           base_container_dir+'segment2.img',
                                           '--ip_scalebar', '100', '--ip', 'aniso', '--ip_range', '0,1', '--ip_colmap', 'new_jet',
                                           '-s', edited_script, '-i', output_file_slurm, input_file_slurm]          

#            print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--ip_scalebar', '100', '-i',output_file, input_file])

        else:
            print 'input not found', input_files[0]


class ImageRange(CombineProcess):
    def __init__(self):
        self.name = 'ImageRange'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_range/']


    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
        

            script_fn = base_script_dir + 'range_borders.txt'

            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]


            input_file = input_files[0]+'.ply.gz'
            output_file = output_files[0]+'.png'
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.png'
        

            return [ 'singularity', 'run', '--app', 'image',
                                           base_container_dir+'segment2.img',  '--ip', 'celltype', '--ip_range', '0,2', '--ip_scalebar', '100',
                                           '-s', edited_script, '-i', output_file_slurm, input_file_slurm]
            


        else:
            print 'input not found', input_files[0]




class AnalyseCells(CombineProcess):
    def __init__(self):
        self.name = 'AnalyseCells'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['cell_data/']


        
    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
        

            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply.gz'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.csv'
        
            return [ 'singularity', 'exec',
                                           base_container_dir+'segment2.img',
                                           'python', '/surface_SPM/src/analyse_cells.py',
                                           '-o', output_file_slurm, input_file_slurm]

        else:
            print 'input not found', input_files[0]
#        open(signal_name, 'a').close()    

#        os.remove(edited_script)
    

class SummaryBladder(Process):
    def __init__(self):
        self.name = 'SummaryBladder'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/','cell_data/']
        self.output_dirs = ['bladder_report/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply.gz'

        if os.path.exists(input_files[0]+'.ply.gz'):
        
            name = os.path.basename(input_files[0])

            output_file = output_files[0]+'.html'

            # Convert ply to ascii

            script_fn = base_script_dir +'convert_binary.txt'

            edited_script = modify_script(script_fn, [input_files[0]+'.ply.gz', output_files[0]+'.ply'])            

            print subprocess.check_output(['python2', headless_local, '-s', edited_script])

            script_fn = base_script_dir +'convert_binary_overlay.txt'

            edited_script = modify_script(script_fn, [input_files[0]+'.ply.gz', output_files[0]+'_overlay.ply'])            

            print subprocess.check_output(['python2', headless_local, '-s', edited_script])

            bladder_report(output_file, name, input_files[0]+'.ply.gz', input_files[1]+'.csv')

        else:
            print 'input not found', input_files[0]
        
        open(signal_name, 'a').close()    
    


class AnalyseAll(object):
    def __init__(self):
        self.name = 'AnalyseAll'
        self.filename_colname = 'filename'
        self.params_colnames = ['DAI','HAI', 'class', 'bladder_length']
        self.input_dirs = ['clean2/']
        self.output_dirs = []

    def run_process(self, record):
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]

        seg_filenames = [ base_data_dir + self.input_dirs[0] + n + '.ply.gz' for n in names ]
        output_filename = base_data_dir + 'report/report.csv'
        
        gland_criterion = '"(norm_area<0.25) or (norm_area>4)"'

        ensure_path_exists(base_data_dir + 'report')

        print subprocess.check_output(['python2', analyseall_local, '-o',output_filename]+seg_filenames)

        extra_filename = base_data_dir + 'report/extended_report.csv'
        
# Prepend name, extra columns to CSV file

        df = pd.read_csv(output_filename)

        extra = [[n]+map(d.get, self.params_colnames) for n, d in zip(names, record.data)]
        
        df2 = pd.DataFrame.from_records(extra, columns = ['name']+self.params_colnames)

        print df2.columns

        df3 = pd.concat([df2, df], axis=1) 

        df3.to_csv(extra_filename, index=False)

class MaxProj(Process):
    def __init__(self):
        self.name = 'MaxProj'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['reduce/']
        self.output_dirs = ['maxproj/']

    def exec_process(self, input_files, output_files, params, signal_name):

        input_file = input_files[0]+'.tif'
        output_file = output_files[0]+'.png'
        maxproj(input_file, output_file)
        open(signal_name, 'a').close()    
        
        
class SummaryTable(object):
    def __init__(self):
        self.name = 'SummaryTable'
        self.filename_colname = 'filename'
        self.params_colnames = ['HAI','DAI','class', 'bladder_length']
        self.input_dirs = ['clean2/']

    def run_process(self, record):
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]
        HAI = [map(d.get, self.params_colnames)[0] for d in record.data]
        DAI = [map(d.get, self.params_colnames)[1] for d in record.data]
        classes = [map(d.get, self.params_colnames)[2] for d in record.data]
        lengths = [map(d.get, self.params_colnames)[3] for d in record.data]
        output_filename = base_data_dir + 'index.html'
        input_filenames = [ base_data_dir + self.input_dirs[0] + n + '.ply.gz' for n in names ]
#        print output_filename, input_filenames

        summary_table(output_filename, base_data_dir, input_filenames, HAI, DAI, classes, lengths, names)



class SummaryGraph(object):
    def __init__(self):
        self.name = 'SummaryGraph'
        self.filename_colname = 'filename'
        self.params_colnames = ['class']
        self.input_dirs = ['clean2/']

    def run_process(self, record):
        output_filename = base_data_dir + 'graph.html'
        input_filename = base_data_dir + 'report/extended_report.csv'
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]
        summary_graph(input_filename, output_filename)



def main():
    process_list = [Enhance(), Project(), MultiSegment(), MultiCombine(),  Clean(), Clip(),  Clean2(), ImageLabels(), ImageRelArea(), ImageAbsLogArea(), ImageAniso(), AnalyseCells(), SummaryBladder(), AnalyseAll(), MaxProj(), SummaryTable(), SummaryGraph()]
    process_dict = dict((p.name,p) for p in process_list)


    if len(sys.argv)>2:
        pipeline = sys.argv[2].split(',')
        if '+' in pipeline[0]:
            all_names = [process.name for process in process_list]
            pipeline = all_names[all_names.index(pipeline[0][:-1]):]
    else:
        pipeline = [process.name for process in process_list]
        
    r = Record()
    r.load_csv(sys.argv[1])

    for pn in pipeline:

        p = process_dict[pn]
        r.add_process_column(p)
        print "Run Process", pn
        p.run_process(r)


    print r.data
    

main()
