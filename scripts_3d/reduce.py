
import sys
import subprocess


fiji = '/usr/users/JIC_a5/fozardj/Downloads/Fiji.app/ImageJ-linux64'

script_path = '/usr/users/JIC_a5/fozardj/bladder_pipeline2/src/reduce/'
script_last = script_path + 'Reduce-last.ijm'
script_single = script_path + 'Reduce.ijm'
script_proj = script_path + 'MaxProj.ijm'


def reduce_file(input_microscopy, output_tif, n=1, m=1, last=False):
    if last:
         out = subprocess.check_output([fiji, '-macro', script_last, input_microscopy+'^'+str(n)+'^'+output_tif])
    else:
         out = subprocess.check_output([fiji, '-macro', script_single, input_microscopy+'^'+str(n)+'^'+str(m)+'^'+output_tif])
    return out

def max_proj_file(input_tif, output_png):
    out = subprocess.check_output([fiji, '-macro', script_proj, input_tif+'^'+output_png])

def main():
    reduce_file(sys.argv[1], sys.argv[2])

if __name__=='__main__':
    main()
    
