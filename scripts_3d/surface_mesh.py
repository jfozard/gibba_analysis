
import sys
import subprocess
import tempfile
import os
from string import Template

python_tool = '/usr/users/JIC_a5/fozardj/meshproject/src/labelled_tool.py'

script_path = '/RG-Stan-Veronica/shared/bladders/3DSeg/'

input_segment_path = "/RG-Stan-Veronica/shared/bladders/3DSeg/segment3d/"
input_enhance_path = "/RG-Stan-Veronica/shared/bladders/3DSeg/enhance/"
input_cropped_path = "/RG-Stan-Veronica/shared/bladders/3DSeg/cropped_stacks/"
input_mask_manual_path = "/RG-Stan-Veronica/shared/bladders/3DSeg/masks/"
input_surface_mask_path = "/RG-Stan-Veronica/shared/bladders/3DSeg/surface_masks/"

output_surface_ply_path = "/RG-Stan-Veronica/shared/bladders/3DSeg/raw_surface/"

input_files = [#'wt80um.tif', 
               #'fixed spiral 010216 slide 3-2.tif',
               #'fixed spiral 020216 slide 1-2.tif',
               #'fixed spiral 020216 slide 1-5.tif', 
               #'wt 160.tif', 
               #'wt 126-2.tif', 
               #'wt122.tif', 
               #'wt 115.tif', 
               #'PI small bladder 3 010316.tif', 
               #'PI small bladder 2 010316.tif',
               'fixed spiral 150216 slide 1-3.tif',
               ]



def modify_script(script, mask1, mask2, outfile):
    fileTemp = tempfile.NamedTemporaryFile(delete = False)
    f = open(script, 'r')
    s = Template(f.read())
    f.close()
    fileTemp.write(s.substitute(dict([('MASK1', mask1), ('MASK2', mask2), ('OUTFILE', outfile)])))
    fileTemp.close()
    return fileTemp.name


def surface_file(input_segment, input_enhance, input_mask, input_surface_mask, output, script='surface_mesh.txt'):
    script = script_path + script

    tmp_script = modify_script(script, input_mask, input_surface_mask, output)


    out = subprocess.check_output(['python', python_tool, '-q', '-r', tmp_script, '-x', input_enhance, input_segment])
    
    os.remove(tmp_script)

#    return out


for fn in input_files:
    surface_file(input_segment_path+fn, input_enhance_path+fn, input_mask_manual_path+fn, input_surface_mask_path+fn, output_surface_ply_path+os.path.splitext(fn)[0]+'.ply')



    
