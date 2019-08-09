
import sys
import csv
import os
import time
import subprocess
import tempfile

from report.summary_table3d import summary_table
from report.summary_table3d_hc import summary_table as summary_table_hc
from report.summary_graph_bokeh import summary_graph
from report.bladder_report import bladder_report
from report.bladder_report_hc import bladder_report as bladder_report_hc
from report.maxproj import maxproj
import pandas as pd

import uuid

local_base_dir = '/home/fozardj/'
local_cwd = local_base_dir+'bladder_pipeline3/'
base_dir = '/run/user/1000/gvfs/smb-share:server=jic-hpc-data,share=hpc-home/' 
slurm_base_dir = '/hpc-home/fozardj/'
signal_dir = base_dir + 'bladder_new/signal/'



base_data_dir = base_dir + 'bladder_new/data3d/'

base_container_dir = slurm_base_dir +'bladder_new/containers/'

base_script_dir = 'scripts/'

slurm_run = local_cwd + base_script_dir + 'slurm_run_signal4.sh'
slurm_run32 = local_cwd + base_script_dir + 'slurm_run_signal32.sh'

temp_script_dir = base_dir + 'bladder_new/scripts/temp/'


input_mask_path = "/run/user/1000/gvfs/smb-share:server=cdbgroup,share=research-groups/Enrico-Coen/COENGROUP/Papers in progess/Claire's Utricularia Paper/Figures/Segmentation_data/Analysis/Data/"
slurm_mask_path = "/jic/research-groups/Enrico-Coen/COENGROUP/Papers in progess/Claire's Utricularia Paper/Figures/Segmentation_data/Analysis/Data/"


local_mask_script = '/home/fozardj/meshproject/src/image_io/convert2.py'

input_mask_fn = '/Slice%03d.png'

timagetk_path_local = base_script_dir + 'segmentation.py'

surfacespm_local = local_base_dir +'surface_spm/src/multi_tool.py'
headless_local = local_base_dir +'surface_spm/src/headless_tool.py'

analysecells_local = local_base_dir +'surface_spm/src/analyse_cells.py'
analyseall_local = local_base_dir +'surface_spm/src/analyse.py'

clip_script_dir = base_script_dir + 'clip3d/'

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

views = ['1_0_0_0_1_0_0_0_1', '0_0_-1_0_1_0_1_0_0', '1_0_0_0_0_-1_0_1_0']

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
        self.input_dirs = ['cropped_stacks/']
        self.output_dirs = ['enhance/']
        self.batch = 1

        
    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
        
            input_files_slurm = [ slurm_base_dir + f[len(base_dir):]+'.tif' for f in input_files]
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.tif'
        
            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]

            print subprocess.check_output([slurm_run,signal_name_slurm,  base_container_dir+'old_pythonspm2.img', '/python-spm/src/enhance/edge.py', "'"+input_files_slurm[0]+"'", "'"+output_file_slurm+"'" ])

        else:
            print 'input not found', input_files
            return False
#        os.remove(edited_script)

class MakeMasks(CombineProcess):
    def __init__(self):
        self.name = 'MakeMasks'
        self.filename_colname = 'filename'
        self.params_colnames = ['mask_dir']
        self.input_dirs = ['orig_stacks_tif/', 'cropped_stacks/']
        self.output_dirs = ['shifted_vvcrop/', 'masks/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.tif'

        mask_full_path = input_mask_path + params['mask_dir']

        if all(os.path.exists(f+'.tif') for f in input_files):


            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in input_files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in output_files]
            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]


            input_mask_slurm = slurm_mask_path+params['mask_dir']+input_mask_fn


            script_fn = local_mask_script
            
            edited_script = modify_script(script_fn, [])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            return ['singularity', 'exec',
                                           base_container_dir+'segment2.img', 'python2', edited_script, '"'+input_files_slurm[0]+'.tif'+'"', '"'+input_files_slurm[1]+'.tif'+'"', '"'+input_mask_slurm+'"', '"'+output_files_slurm[0]+'"', '"'+output_files_slurm[1]+'"']
            
        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()

class SPM3D(CombineProcess):
    def __init__(self):
        self.name = 'SPM'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['enhance/']
        self.output_dirs = ['spm_bdd/', 'spm_labels/']
        self.batch = 1

        
    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
        
            input_files_slurm = [ '"' +slurm_base_dir + f[len(base_dir):]+'.tif'+'"' for f in input_files]
            output_files_slurm = [ '"'+slurm_base_dir + f[len(base_dir):]+'.tif'+'"' for f in output_files]
        
            return ['singularity', 'exec', base_container_dir+'old_pythonspm2.img', 'python', '/python-spm/src/new_single.py', input_files_slurm[0], output_files_slurm[0], output_files_slurm[1] ]

        else:
            print 'input not found', input_files[0]

#        os.remove(edited_script)

class Segment3D(CombineProcess):
    def __init__(self):
        self.name = 'Segment'
        self.filename_colname = 'filename'
        self.params_colnames = ['seg3d_level']
        self.input_dirs = ['spm_bdd/']
        self.output_dirs = ['segment3d/']
        self.batch = 1

        
    def get_slurm_command(self, input_files, output_files, params,):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
            s = params['seg3d_level']
            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in input_files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in output_files]
            return ['singularity', 'exec', base_container_dir+'old_pythonspm.img', 'python', '/python-spm/src/seg3d_f.py', input_files_slurm[0], output_files_slurm[0], str(s) ]

#            print subprocess.check_output([base_dir+'slurm_run_signal.sh', signal_name_slurm, base_container_dir+'pythonspm.img', '/python-spm/src/seg3d_f.py', "'"+input_files_slurm[0]+"'", "'"+output_files_slurm[0]+"'"])

        
        else:
            print 'input not found', input_files[0]

"""
class SurfaceMask(Process):
    def __init__(self):
        self.name = 'SurfaceMask'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['enhance/']
        self.output_dirs = ['surface_masks/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
            script_fn = base_script_dir + 'surfacemask_script.txt'
        

            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in input_files]
            output_files_slurm = [ slurm_base_dir + f[len(base_dir):]+'.tif' for f in output_files]

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
        
            edited_script = modify_script(script_fn, [output_files_slurm[0]])
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            print input_files_slurm, output_files

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
    
            print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'meshproject.img', '-r', edited_script, '-i', input_files_slurm[0]])

        else:
            print 'input not found', input_files[0]
            open(signal_name, 'a').close()
"""

"""
class ExtractSurface(Process):
    def __init__(self):
        self.name = 'ExtractSurface'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['segment3d/','masks/', 'surface_masks/', 'enhance/', 'cropped_stacks/']
        self.output_dirs = ['segment/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        
            script_fn = base_script_dir + 'extract_surface_script.txt'
        

            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in input_files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.ply'+"'" for f in output_files]

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
        

            edited_script = modify_script(script_fn, input_files_slurm + output_files_slurm)
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            print input_files_slurm, output_files

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
        
            print subprocess.check_output([base_dir+'slurm_run_signal.sh', signal_name_slurm, base_container_dir+'meshproject.img', '-r', edited_script])


        else:
            print 'input not found', input_files[0], [ (f,os.path.exists(f+'.tif')) for f in input_files]
            open(signal_name, 'a').close()


class Clip(Process):
    def __init__(self):
        self.name = 'Clip'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['segment/']
        self.output_dirs = ['clipped/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):

            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply'


            script_fn = clip_script_dir + 'clip-'+os.path.basename(input_files[0])+'.txt'
            
            edited_script = modify_script(script_fn, [] )

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            print edited_script

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]
        
    
            print subprocess.check_output([base_dir+'slurm_run_signal.sh', signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script, '-o', "'"+output_file_slurm+"'", "'"+input_file_slurm+"'"])


        else:
            print 'input not found', input_files[0]
            open(signal_name, 'a').close()

class Clean2(Process):
    def __init__(self):
        self.name = 'Clean2'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['clipped/']
        self.output_dirs = ['clean2/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
            script_fn = base_script_dir + 'clean2.txt'
            

            edited_script = modify_script(script_fn, [])
        
            print edited_script

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply'
        
            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]

            print subprocess.check_output([base_dir+'slurm_run_signal.sh', signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script, '-o', "'"+output_file_slurm+"'", "'"+input_file_slurm+"'"])



        else:
            print 'input not found', input_files[0]
            open(signal_name, 'a').close()




class ImageLabels(Process):
    def __init__(self):
        self.name = 'ImageLabels'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_labels/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
#            script_fn = base_script_dir + 'segment_script.txt'
            
#            print  params['Segment_nbhoodsize']

#            edited_script = modify_script(script_fn, [params['Segment_nbhoodsize']])
        
#            print edited_script

#            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        

            script_fn = base_script_dir + 'borders.txt'


            input_file = input_files[0]+'.ply'
            

            for i, m in enumerate(views):
                if i>0:
                    output_file = output_files[0]+'-'+str(i)+'.png'
                else:
                    output_file = output_files[0]+'.png'
            
                    
                print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--ip_scalebar', '100', '--i_matrix', m, '-i',output_file, input_file])
#            print ['python2', surfacespm_local, '-i',output_file, input_file]


        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()

class ImageRange(Process):
    def __init__(self):
        self.name = 'ImageRange'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_range/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
            script_fn = base_script_dir + 'range_borders.txt'


            input_file = input_files[0]+'.ply'
            output_file = output_files[0]+'.png'
        
            
            print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--ip', 'celltype', '--ip_range', '0,2', '--ip_scalebar', '100', '-i',output_file, input_file])
#            print ['python2', surfacespm_local, '-i',output_file, input_file]


        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)
        open(signal_name, 'a').close()    
        


class ImageRelArea(Process):
    def __init__(self):
        self.name = 'ImageRelArea'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_relarea/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
#            script_fn = base_script_dir + 'segment_script.txt'
            
#            print  params['Segment_nbhoodsize']

#            edited_script = modify_script(script_fn, [params['Segment_nbhoodsize']])
        
#            print edited_script

#            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            script_fn = base_script_dir + 'borders.txt'

            input_file = input_files[0]+'.ply'
            output_file = output_files[0]+'.png'
        
        
            for i, m in enumerate(views):
                if i>0:
                    output_file = output_files[0]+'-'+str(i)+'.png'
                else:
                    output_file = output_files[0]+'.png'
            
            

                print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--ip_scalebar', '100', '--i_matrix', m, '--ip', 'area', '-i',output_file, input_file])
#            print ['python2', surfacespm_local, '-i',output_file, input_file]


        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()

class ImageAbsLogArea(Process):
    def __init__(self):
        self.name = 'ImageAbsLogArea'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_abslogarea/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
#            script_fn = base_script_dir + 'segment_script.txt'
            
#            print  params['Segment_nbhoodsize']

#            edited_script = modify_script(script_fn, [params['Segment_nbhoodsize']])
        
#            print edited_script

#            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            input_file = input_files[0]+'.ply'
            output_file = output_files[0]+'.png'

            script_fn = base_script_dir + 'borders.txt'  

      
        
            for i, m in enumerate(views):
                if i>0:
                    output_file = output_files[0]+'-'+str(i)+'.png'
                else:
                    output_file = output_files[0]+'.png'
            
            

                print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--i_matrix', m, '--ip_scalebar', '100', '--ip', 'area', '--ip_range', '25,2500', '--ip_log', '-i',output_file, input_file])
#            print ['python2', surfacespm_local, '-i',output_file, input_file]


        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()
#        os.remove(edited_script)

class ImageAniso(Process):
    def __init__(self):
        self.name = 'ImageAniso'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['image_aniso/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
#            script_fn = base_script_dir + 'segment_script.txt'
            
#            print  params['Segment_nbhoodsize']

#            edited_script = modify_script(script_fn, [params['Segment_nbhoodsize']])
        
#            print edited_script

#            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            input_file = input_files[0]+'.ply'
            output_file = output_files[0]+'.png'

            script_fn = base_script_dir + 'aniso_vectors.txt'
        
            for i, m in enumerate(views):
                if i>0:
                    output_file = output_files[0]+'-'+str(i)+'.png'
                else:
                    output_file = output_files[0]+'.png'
            

                print subprocess.check_output(['python2', surfacespm_local, '-s', script_fn, '--i_matrix', m, '--ip_scalebar', '100', '--ip', 'aniso', '-i',output_file, input_file])
#            print ['python2', surfacespm_local, '-i',output_file, input_file]


        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()
#        os.remove(edited_script)
        

class AnalyseCells(Process):
    def __init__(self):
        self.name = 'AnalyseCells'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/']
        self.output_dirs = ['cell_data/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
        
            input_file = input_files[0]+'.ply'
            output_file = output_files[0]+'.csv'
        
            print subprocess.check_output(['python2', analysecells_local, '-o', output_file, input_file])
#            print ['python2', surfacespm_local, '-i',output_file, input_file]


        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()
#        os.remove(edited_script)
    

class SummaryBladder(Process):
    def __init__(self):
        self.name = 'SummaryBladder'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['clean2/','cell_data/']
        self.output_dirs = ['bladder_report/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
            name = os.path.basename(input_files[0])

            output_file = output_files[0]+'.html'

            # Convert ply to ascii

            script_fn = base_script_dir +'convert_ascii.txt'


            edited_script = modify_script(script_fn, [input_files[0]+'.ply', output_files[0]+'.ply'])            

            print subprocess.check_output(['python2', headless_local, '-s', edited_script])

            script_fn = base_script_dir +'convert_ascii_overlay.txt'

            edited_script = modify_script(script_fn, [input_files[0]+'.ply', output_files[0]+'_overlay.ply'])

            print subprocess.check_output(['python2', headless_local, '-s', edited_script])

            bladder_report(output_file, name, input_files[0]+'.ply', input_files[1]+'.csv')

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

        seg_filenames = [ base_data_dir + self.input_dirs[0] + n + '.ply' for n in names ]
        output_filename = base_data_dir + 'report/report.csv'
        
        print subprocess.check_output(['python2', analyseall_local, '-o',output_filename]+seg_filenames)

        extra_filename = base_data_dir + 'report/extended_report.csv'
        
# Prepend name, extra columns to CSV file

        df = pd.read_csv(output_filename)

        extra = [[n]+map(d.get, self.params_colnames) for n, d in zip(names, record.data)]
        
        df2 = pd.DataFrame.from_records(extra, columns = ['name']+self.params_colnames)

        print df2.columns

        df3 = pd.concat([df2, df], axis=1) 

        df3.to_csv(extra_filename, index=False)

class SummaryTable(object):
    def __init__(self):
        self.name = 'SummaryTable'
        self.filename_colname = 'filename'
        self.params_colnames = ['DAI','HAI','class', 'bladder_length']
        self.input_dirs = ['clean2/']

    def run_process(self, record):
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]
        HAI = [map(d.get, self.params_colnames)[0] for d in record.data]
        DAI = [map(d.get, self.params_colnames)[1] for d in record.data]
        classes = [map(d.get, self.params_colnames)[2] for d in record.data]
        lengths = [map(d.get, self.params_colnames)[3] for d in record.data]
        output_filename = base_data_dir + 'index.html'
        input_filenames = [ base_data_dir + self.input_dirs[0] + n + '.ply' for n in names ]
#        print output_filename, input_filenames

        summary_table(output_filename, base_data_dir, input_filenames, DAI, HAI, classes, lengths, names)

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

"""




class ExtractHCSurface(CombineProcess):
    def __init__(self):
        self.name = 'ExtractHCSurface'
        self.filename_colname = 'filename'
        self.params_colnames = ['surface_erode']
        self.input_dirs = ['segment3d/','hand_cleaned3d/', 'enhance/', 'cropped_stacks/']
        self.output_dirs = ['hc_segment/']
        self.batch=1
        
    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        

            print params['surface_erode']

            script_fn = base_script_dir + 'extract_hc_surface_script'+params['surface_erode']+'.txt'
        

            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in input_files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.ply'+"'" for f in output_files]


            edited_script = modify_script(script_fn, input_files_slurm + output_files_slurm)
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            print input_files_slurm, output_files

            return [base_container_dir+'meshproject.img', '-r', edited_script]

        else:
            print 'input not found', input_files[0], [ (f,os.path.exists(f+'.tif')) for f in input_files]
#        os.remove(edited_script)



class TImagetk(CombineProcess):
    def __init__(self):
        self.name = 'TImagetk'
        self.filename_colname = 'filename'
        self.params_colnames = ['timagetk_threshold', 'timagetk_blur']
        self.input_dirs = ['cropped_stacks/']
        self.output_dirs = ['timagetk/']
        self.batch = 1
        
    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.tif'

        if all(os.path.exists(f+'.tif') for f in input_files):
        

            print params['timagetk_threshold']

            script_fn = timagetk_path_local
            
            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in input_files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.tif'+"'" for f in output_files]

            edited_script = modify_script(script_fn, {})
            edited_script = slurm_base_dir + edited_script[len(base_dir):]


            print input_files_slurm, output_files_slurm, edited_script

        

            return [ base_container_dir+'timagetk.img', 'python', edited_script, input_files_slurm[0], output_files_slurm[0], params['timagetk_threshold'], params['timagetk_blur'] ]

        else:
            print 'input not found', input_files[0], [ (f,os.path.exists(f+'.tif')) for f in input_files]
#        os.remove(edited_script)

class ExtractTKSurface(CombineProcess):
    def __init__(self):
        self.name = 'ExtractTKSurface'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['timagetk/', 'enhance/', 'cropped_stacks/','hc_segment/']
        self.output_dirs = ['tk_surface/']
        self.batch = 1
        
    def get_slurm_command(self, input_files, output_files, params):


        files = [f+'.tif' for f in input_files[:3]] + [f+'.ply' for f in input_files[3:]]
        print files
        if all(os.path.exists(f) for f in files):
        
            script_fn = base_script_dir + 'extract_seg.txt'
        

            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+"'" for f in files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.ply'+"'" for f in output_files]


            edited_script = modify_script(script_fn, input_files_slurm + output_files_slurm)
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            print input_files_slurm, output_files

            return [base_container_dir+'meshproject.img', '-r', edited_script]

        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)



class ExtractSPMSurface(CombineProcess):
    def __init__(self):
        self.name = 'ExtractSPMSurface'
        self.filename_colname = 'filename'
        self.params_colnames = ['spm_path']
        self.input_dirs = ['enhance/', 'cropped_stacks/','hc_segment/']
        self.output_dirs = ['spm_surface/']

        
    def get_slurm_command(self, input_files, output_files, params):


        files = [f+'.tif' for f in input_files[:1]] + [base_data_dir+params['spm_path']] + [f+'.tif' for f in input_files[1:2]] + [f+'.ply' for f in input_files[2:]]
        print files
        if os.path.exists(files[0])  and all(os.path.exists(f) for f in files[2:]) and params['spm_path']:
        
            print files
            script_fn = base_script_dir + 'extract_seg_spm.txt'
        

            input_files_slurm = [ "'" +slurm_base_dir + f[len(base_dir):]+"'" for f in files]
            output_files_slurm = [ "'"+slurm_base_dir + f[len(base_dir):]+'.ply'+"'" for f in output_files]


            edited_script = modify_script(script_fn, input_files_slurm + output_files_slurm)
            edited_script = slurm_base_dir + edited_script[len(base_dir):]

            print input_files_slurm, output_files
        

            return  [ base_container_dir+'meshproject.img', '-r', edited_script ]

        else:
            print 'input not found', input_files[0]
            return []
#        os.remove(edited_script)



"""
class MultiSegment(Process):
    def __init__(self):
        self.name = 'MultiSegment'
        self.filename_colname = 'filename'
        self.params_colnames = ['Segment_nbhoodsize', 'Segment_mean', 'Segment_grad']
        self.input_dirs = ['hc_segment/']
        self.output_dirs = ['multisegment/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        all_signals = []
        if os.path.exists(input_files[0]+'.ply'):
        
            for seed in range(5):
                script_fn = base_script_dir + 'multisegment_script_hc.txt'
            
                print  params['Segment_nbhoodsize'], params['Segment_mean'], params['Segment_grad']
                

                edited_script = modify_script(script_fn, [seed, seed, seed, params['Segment_mean'], params['Segment_nbhoodsize'], params['Segment_grad']])
        
                print edited_script

                edited_script = slurm_base_dir + edited_script[len(base_dir):]
                
                input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply'
                output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.'+str(seed)+'.ply'
        

                signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]+str(seed)
                all_signals.append(signal_name+str(seed))
                print subprocess.check_output([slurm_run, signal_name_slurm, base_container_dir+'segment.img', '-s', edited_script, '-o', '"'+output_file_slurm+'"', '"'+input_file_slurm+'"'])


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
        print input_files[0]+'.ply'

        if_list = [input_files[0]+'.'+str(i)+'.ply' for i in range(5)]
        if all(os.path.exists(f) for f in if_list):
        
            input_files_slurm = [slurm_base_dir + f[len(base_dir):] for f in if_list]
            input_files_slurm = [ '"'+s+'"' for s in input_files_slurm ]

            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply'
        
            print input_files_slurm

            print 'Combine level', params['Combine_level']

            signal_name_slurm = slurm_base_dir + signal_name[len(base_dir):]

            print subprocess.check_output([slurm_run, signal_name_slurm, 'singularity', 'run', '--app', 'combine', base_container_dir+'segment.img', params['Combine_level'], '"'+output_file_slurm+'"'] + input_files_slurm)

        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)
"""

class HCClip(CombineProcess):
    def __init__(self):
        self.name = 'HCClip'
        self.filename_colname = 'filename'
        self.params_colnames = ['best_seg']
        self.input_dirs = ['hc_segment/','tk_surface/','spm_surface/']
        self.output_dirs = ['hc_clipped/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):

            idx = int(params['best_seg'])
            input_file_slurm = slurm_base_dir + input_files[idx][len(base_dir):]+'.ply'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply'


            script_fn = clip_script_dir + 'clip-'+os.path.basename(input_files[idx])+'.txt'
            
            edited_script = modify_script(script_fn, [] )

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            print edited_script
        
            return [base_container_dir+'segment.img', '-s', edited_script, '-o', "'"+output_file_slurm+"'", "'"+input_file_slurm+"'"]


        else:
            print 'input not found', input_files[0]
#        os.remove(edited_script)

class HCClean2(CombineProcess):
    def __init__(self):
        self.name = 'HCClean2'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['hc_clipped/']
        self.output_dirs = ['hc_clean2/']

    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
            script_fn = base_script_dir + 'clean2.txt'
            

            edited_script = modify_script(script_fn, [])
        
            print edited_script

            edited_script = slurm_base_dir + edited_script[len(base_dir):]
        
            input_file_slurm = slurm_base_dir + input_files[0][len(base_dir):]+'.ply'
            output_file_slurm = slurm_base_dir + output_files[0][len(base_dir):]+'.ply'
        
            return [base_container_dir+'segment.img', '-s', edited_script, '-o', "'"+output_file_slurm+"'", "'"+input_file_slurm+"'"]
            

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
        
class HCImageLabels(ImageProcess):
    def __init__(self):
        self.name = 'HCImageLabels'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['hc_image_labels/']
        self.script_fn = base_script_dir + 'borders.txt'
        self.extra_options = []
        self.batch = 1

        
class HCImageRelArea(ImageProcess):
    def __init__(self):
        self.name = 'HCImageRelArea'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['hc_image_relarea/']
        self.script_fn = base_script_dir + 'borders.txt'
        self.extra_options = ['--ip', 'area', '--ip_colmap', 'new_jet']
        self.batch = 1

            

class HCImageAbsLogArea(ImageProcess):
    def __init__(self):
        self.name = 'HCImageAbsLogArea'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['hc_image_abslogarea/']
        self.script_fn = base_script_dir + 'borders.txt'

        self.extra_options =  [ '--ip', 'area', '--ip_range', '25,2500', '--ip_log' , '--ip_colmap', 'new_jet']
        self.batch = 1




class HCImageAniso(ImageProcess):
    def __init__(self):
        self.name = 'HCImageAniso'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['hc_image_aniso/']
        self.script_fn = base_script_dir + 'aniso_vectors.txt'
        self.extra_options =  [ '--ip', 'aniso', '--ip_range', '0,1', '--ip_colmap', 'new_jet' ]
        self.batch = 1

        



class HCImageRange(ImageProcess):
    def __init__(self):
        self.name = 'HCImageRange'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['hc_image_range/']
        self.script_fn = base_script_dir + 'range_borders.txt'
        self.extra_options =  [ '--ip', 'celltype', '--ip_range', '0,2', '--ip_colmap', 'new_jet' ]
        self.batch = 1

class HCAnalyseCells(CombineProcess):
    def __init__(self):
        self.name = 'HCAnalyseCells'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['hc_cell_data/']

        
    def get_slurm_command(self, input_files, output_files, params):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        

            input_file_slurm = '"'+slurm_base_dir + input_files[0][len(base_dir):]+'.ply'+'"'
            output_file_slurm = '"'+slurm_base_dir + output_files[0][len(base_dir):]+'.csv'+'"'
        
            return ['singularity', 'exec',
                                           base_container_dir+'segment2.img',
                                           'python', '/surface_SPM/src/analyse_cells.py',
                                           '-o', output_file_slurm, input_file_slurm]


class MaxProj(Process):
    def __init__(self):
        self.name = 'MaxProj'
        self.filename_colname = 'filename'
        self.params_colnames = []
        self.input_dirs = ['cropped_stacks/']
        self.output_dirs = ['maxproj/']

    def exec_process(self, input_files, output_files, params, signal_name):

        input_file = input_files[0]+'.tif'
        output_file = output_files[0]+'.png'
        maxproj(input_file, output_file)
        open(signal_name, 'a').close()    



class HCSummaryBladder(Process):
    def __init__(self):
        self.name = 'HCSummaryBladder'
        self.filename_colname = 'filename'
        self.params_colnames = ['']
        self.input_dirs = ['hc_clean2/','hc_cell_data/']
        self.output_dirs = ['hc_bladder_report/']

    def exec_process(self, input_files, output_files, params, signal_name):
        print input_files[0]+'.ply'

        if os.path.exists(input_files[0]+'.ply'):
        
            name = os.path.basename(input_files[0])

            output_file = output_files[0]+'.html'

            # Convert ply to ascii

            script_fn = base_script_dir +'convert_ascii.txt'

            edited_script = modify_script(script_fn, [input_files[0]+'.ply', output_files[0]+'.ply'])            

            print subprocess.check_output(['python2', headless_local, '-s', edited_script])

            script_fn = base_script_dir +'convert_ascii_overlay.txt'

            edited_script = modify_script(script_fn, [input_files[0]+'.ply', output_files[0]+'_overlay.ply'])            

            print subprocess.check_output(['python2', headless_local, '-s', edited_script])

            bladder_report_hc(output_file, name, input_files[0]+'.ply', input_files[1]+'.csv')

        else:
            print 'input not found', input_files[0]
        open(signal_name, 'a').close()
    


class HCAnalyseAll(object):
    def __init__(self):
        self.name = 'HCAnalyseAll'
        self.filename_colname = 'filename'
        self.params_colnames = ['class', 'bladder_length']
        self.input_dirs = ['hc_clean2/']
        self.output_dirs = ['']

    def run_process(self, record):
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]


        directory = base_data_dir + 'report/'
        if not os.path.exists(directory):
                os.makedirs(directory)

        
        seg_filenames = [ base_data_dir + self.input_dirs[0] + n + '.ply' for n in names ]
        output_filename = base_data_dir + 'report/hc_report.csv'
        
        print subprocess.check_output(['python2', analyseall_local, '-o',output_filename]+seg_filenames)

        extra_filename = base_data_dir + 'report/hc_extended_report.csv'
        
# Prepend name, extra columns to CSV file

        df = pd.read_csv(output_filename)

        extra = [[n]+map(d.get, self.params_colnames) for n, d in zip(names, record.data)]
        
        df2 = pd.DataFrame.from_records(extra, columns = ['name']+self.params_colnames)

        print df2.columns

        df3 = pd.concat([df2, df], axis=1) 

        df3.to_csv(extra_filename, index=False)

class HCSummaryTable(object):
    def __init__(self):
        self.name = 'HCSummaryTable'
        self.filename_colname = 'filename'
        self.params_colnames = ['DAI','HAI','class', 'trap_length']
        self.input_dirs = ['hc_clean2/']

    def run_process(self, record):
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]
        DAI = [map(d.get, self.params_colnames)[0] for d in record.data]
        HAI = [map(d.get, self.params_colnames)[1] for d in record.data]
        classes = [map(d.get, self.params_colnames)[2] for d in record.data]
        lengths = [map(d.get, self.params_colnames)[3] for d in record.data]
        output_filename = base_data_dir + 'hc_index.html'
        input_filenames = [ base_data_dir + self.input_dirs[0] + n + '.ply' for n in names ]
#        print output_filename, input_filenames

        summary_table_hc(output_filename, base_data_dir, input_filenames, DAI, HAI,  classes, lengths, names)

class HCSummaryGraph(object):
    def __init__(self):
        self.name = 'HCSummaryGraph'
        self.filename_colname = 'filename'
        self.params_colnames = ['class']
        self.input_dirs = ['hc_clean2/']

    def run_process(self, record):
        output_filename = base_data_dir + 'hc_graph.html'
        input_filename = base_data_dir + 'report/hc_extended_report.csv'
        names = [os.path.splitext(d.get(self.filename_colname))[0] for d in record.data]
        summary_graph(input_filename, output_filename)







def main():
    process_list = [Enhance(), MakeMasks(), SPM3D(), Segment3D(), #SurfaceMask(),
                    #ExtractSurface(), Clip(), Clean2(), ImageLabels(), ImageRelArea(), ImageAbsLogArea(), ImageAniso(), AnalyseCells(), SummaryBladder(), AnalyseAll(), SummaryTable(), SummaryGraph(),
                    ExtractHCSurface(), TImagetk(), ExtractTKSurface(), ExtractSPMSurface(),
                    #MultiSegment(), MultiCombine(),
                    HCClip(), HCClean2(), HCImageLabels(), HCImageRelArea(), HCImageAbsLogArea(), HCImageAniso(), HCImageRange(), HCAnalyseCells(), HCSummaryBladder(), HCAnalyseAll(), MaxProj(), HCSummaryTable(), HCSummaryGraph()]
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
