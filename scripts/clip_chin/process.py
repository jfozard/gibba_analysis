
import os
import os.path
import sys

f = open(sys.argv[1], 'r')
d = f.readlines()
d = [l for l in d if ('add_ply' not in l) and ('save_ply' not in l) and ('select_obj' not in l) and ('flip_xy' not in l)]
g = open('clip'+os.path.basename(sys.argv[1])[3:], 'w')
g.write("[\"add_ply\",('%s',),{'orientation':None}]\n")
g.writelines(d)
g.write("[\"rotate_axis\",(array([0., 0., 1.]), 1.5707963267948966, False),{}]\n")
g.write("[\"save_ply\",('%s',),{}]")
