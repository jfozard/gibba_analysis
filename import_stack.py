import sys
import numpy as np
import os
import subprocess

fiji = '/usr/users/JIC_a5/fozardj/Downloads/Fiji.app/ImageJ-linux64'
image_server_path = '/run/user/11258/gvfs/smb-share:server=jic-image-data,share=jic-image_database' 
image_server_windows_path = r'\\jic-image-data\jic-image_database'

def convert_local_path(path):
   if path.startswith(image_server_windows_path):
       new_path = image_server_path + path[len(image_server_windows_path):].replace('\\','/')
       return new_path
   else:
       return None

script_path = 'scripts/'
script_last = script_path + 'Reduce-last.ijm'
script_single = script_path + 'Reduce.ijm'

def reduce_file(input_microscopy, output_tif, n=1, m=1, last=False):
    if last:
         out = subprocess.check_output([fiji, '-macro', script_last, input_microscopy+'^'+str(n)+'^'+output_tif])
    else:
         out = subprocess.check_output([fiji, '-macro', script_single, input_microscopy+'^'+str(n)+'^'+str(m)+'^'+output_tif])
    return out


def main():
    reduce_file(convert_local_path(sys.argv[1]), sys.argv[2], 1 if len(sys.argv)<=3 else int(sys.argv[3]))
    

if __name__=="__main__":
    main()

