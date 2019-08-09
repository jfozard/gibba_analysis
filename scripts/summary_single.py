

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

from projection import proj
from image_io.import_lif import load_lif_stack, stop_javabridge
from math import sqrt, pi
import skimage.exposure
import scipy.ndimage as nd
from graphs.heatmap import heatmap, heatmap_array, heatmap_array_shrunk
from graphs.view_segmentation import view_segmentation_lines
#from graphs.cell_size_histogram import cell_size_histogram
from segmentation.spm import SPM
from skimage.segmentation import relabel_sequential

from PIL import Image

import pandas as pd

import SimpleITK as sitk

from analysis.cell_size_shape import analyse_cells

mc = colorConverter.to_rgba('blue')

# make the colormaps
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',[mc, mc],256)

cmap._init() # create the _lut array, with rgba values

alphas = np.linspace(0, 1, cmap.N+3)
cmap._lut[:,-1] = alphas


def wshed2d(im, level=20, min_area=None):
    itk_image = sitk.GetImageFromArray(im)
    gaussian_filter = sitk.DiscreteGaussianImageFilter()
    gaussian_filter.SetVariance(2.0)
    itk_image = gaussian_filter.Execute(itk_image)
    itk_image = sitk.MorphologicalWatershed(itk_image, level=level, markWatershedLine=False, fullyConnected=False)    
    if min_area:
        itk_image = sitk.RelabelComponent(itk_image, min_area)
    else:
        itk_image = sitk.RelabelComponent(itk_image)
    itk_image = sitk.Mask(itk_image, itk_image!=1)
    itk_image = sitk.RelabelComponent(itk_image)
    return sitk.GetArrayFromImage(itk_image)
    

def vis2d(ma, sps, indices, axis=1, aspect=1, threshold=200, color='r'):
    N = len(indices)
    plt.figure(figsize=(15, 2*N))
    other_axis = 3-axis
    for i, idx in enumerate(indices):
        s = np.take(ma, idx, axis)
        h = np.take(sps, idx, axis-1)
        if True:
            plt.imshow(np.minimum(threshold, s), cmap=plt.cm.gray, aspect=1.0/aspect)
            plt.hold(True)
            print s.shape, ma.shape, h.shape
            plt.plot(range(ma.shape[other_axis]), h, color, lw=1)
            plt.ylim(ma.shape[0], 0)
            plt.xlim(0, ma.shape[other_axis])
            plt.xticks([])
#            if i>0:
            plt.yticks([])


def write_png(data, filename):
    im = Image.fromarray(data)
    im.save(filename)


def remove_small_large(cells, min_px, max_px):
    bc = np.bincount(cells.flatten())
    for i in range(1, len(bc)):
        if bc[i]<min_px or bc[i]>max_px:
            cells[cells==i]=0
    return relabel_sequential(cells)[0]


def split_disconnected(array):
    labels = np.unique(array)[1:]
    max_label = max(labels)
    for l in labels:
        nl, v = nd.label(array==l)
        if v>1:
            for i in range(1, v):
                max_label+=1
                array[nl==i] = max_label
    

def run_spm(array):
    s = SPM(array, None)
    s.Run(verbose=False)

    cells = s.state

    split_disconnected(cells)
    # find median cell size

    sizes = np.bincount(cells.flatten())[1:]
    sizes = sizes[sizes!=0]
    median_size = np.median(sizes)

    cells = remove_small_large(cells, 0.1*median_size, 1e60)

    distances, indices = nd.distance_transform_edt(cells==0, return_indices=True)
    
    new_cells = cells[tuple(indices)]
    new_cells = remove_small_large(new_cells, 0.2*median_size, 10*median_size)

    return new_cells

def process_array(ma, spacing, title='', output_dir='', do_spm=1):
    """
    Args:
        ma (np.ndarray) - Image data array.
        spacing (tuple) - Voxel spacing in pixels - in order ZYX.
    
    Returns:
        surf (np.ndarray) - Position of the projection surface at each 
                            image point (float)
        image (np.ndarray) - Projection of the stack onto the surface
        cells (np.ndarray) - Segmented cells (unsigned short labels)
                             0=background, cells labelled beginning at 1
    """

    """
    Calculate region in x-y plane occupied by bladder - use this
    to estimate lengthscale
    """

    sns.set_style('white')
    sns.set_context('talk')

    f = open(output_dir + 'summary.html', 'w')

    pp = np.max(ma, axis=0)
    pp = nd.gaussian_filter(pp, 1.0)
    mask2d = pp>0.1*np.mean(pp)
    ci, cj = nd.center_of_mass(mask2d)

    area = spacing[1]*spacing[2]*np.sum(mask2d)

    f.write("""<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title> {} </title>
    </head>
    <body>
    """.format(title))

    if title:
        f.write("<h2> {} </h2>\n".format(title))
    f.write("<h3>Image summary</h3>\n")
    f.write("<p>Image spacing {} {} {} microns (x, y, z directions)</p>\n".format(spacing[2], spacing[1], spacing[0]))

    print "area " +str(area)+"\n"
    f.write("<p>Bladder cross-sectional (xy) area {} microns^2 = {} px^2 </p>\n".format(area, np.sum(mask2d)))

    print "equivalent diameter", sqrt(area/pi*4.0)
    f.write("<p>Equivalent diameter {} microns</p>\n".format(sqrt(area/pi*4.0)))



    """
    First task - determine the (smooth) projection surface
    Options here are whether to use max projection or threshold,
    and then how smooth to require the surface to be!
    
    Suspect the number of cells on the flanks of the bladder is roughly
    constant, which dictates the lengthscale

    Want to smooth in the x- and y- directions by about half the cell size...

    """

    cell_size = sqrt(area/4000.0)
    
    cell_px = cell_size/spacing[1]

    print "estimated cell_size", cell_size
    print "estimated cell_px", cell_px
    f.write("<p>Estimated cell diameter (used to smooth surface) {} microns = {} px</p>\n".format(cell_size, cell_px))



    f.write("<h4> Bladder size estimation </h4>\n")
    max_proj = np.max(ma, axis=0)
    plt.figure(figsize=(12,12))
    plt.imshow(max_proj, cmap=plt.cm.gray)
    plt.savefig(output_dir+'max_proj.png', bbox_inches='tight')    

    plt.figure()
    plt.imshow(max_proj, cmap=plt.cm.gray)
    m2 = mask2d ^ nd.binary_erosion(mask2d, iterations=2)
    plt.imshow(m2, cmap=cmap)
    circle = plt.Circle((cj, ci), sqrt(np.sum(mask2d)/pi), color='r',
                        fill=False)
    plt.gca().add_artist(circle)
    plt.axis('off')
    plt.savefig(output_dir+'bladder_size.png', bbox_inches='tight')    
    f.write('<img src="bladder_size.png">\n')     
    

    bl1_scale = [0.3*cell_size/spacing[0], 0.3*cell_px, 0.3*cell_px]

    bl1 = nd.gaussian_filter(ma, bl1_scale)
    ps = proj.threshold_indices_z(bl1, 1, 5)
    sps = nd.gaussian_filter(ps.astype(float), 1.0*cell_px)

    """
    Second task - plot projection surface on top of the stack to validate 
    choice of surface (important as sometimes doesn't work).
    """
 
    plt.figure()
    plt.imshow(sps)
    plt.hold(True)
    plt.plot((cj, cj), (0, sps.shape[0]), 'b-')
    plt.plot((0, sps.shape[1]), (ci, ci), 'r-')

    plt.axis('off')
    plt.savefig(output_dir+'projection_surface.png', bbox_inches='tight')

    f.write('<h3> Projection surface </h3>') 
    f.write('<img src="projection_surface.png">\n') 

#    write_png((255.0*sps/np.max(sps)).astype(np.uint8), 
 #             output_dir+'projection_surface.png')

    vis2d(ma, sps, [ci], axis=1, aspect=spacing[1]/spacing[0])
    plt.savefig(output_dir+"projection_surface_x.png", bbox_inches='tight')
    f.write('<h4> Projection cross-section (x-direction) </h4>') 
    f.write('<img src="projection_surface_x.png">\n') 

    vis2d(ma, sps, [cj], axis=2, color='b', aspect=spacing[1]/spacing[0])
    plt.savefig(output_dir+'projection_surface_y.png',  bbox_inches='tight')
    f.write('<h4> Projection cross-section (y-direction) </h4>') 
    f.write('<img src="projection_surface_y.png">\n') 


    """
    Third task - project stack onto surface
    """

    bl = nd.gaussian_filter(ma, 0.5)
    proj_signal = proj.projection_from_surface_normal_z(bl, sps, dm=-5, dp=8)
#    proj_signal = proj.projection_from_surface_z(bl, sps, dp=20)

    write_png(proj_signal, output_dir+'projected_signal.png')
    f.write('<h3> Projected signal </h3>') 
    f.write('<img src="projected_signal.png">\n') 

    """
    Fourth task - CLAHE on image
    """
    norm_signal = (255*skimage.exposure.equalize_adapthist(proj_signal, int(0.5*cell_px))).astype('uint8')
    write_png(norm_signal, output_dir+'normalized_signal.png')
    f.write('<h3> Locally normalized signal </h3>') 
    f.write('<img src="normalized_signal.png">\n') 

    plt.figure()
    plt.imshow(norm_signal, cmap=plt.cm.gray)
    
    """
    Fifth task - segment cells
    """
    min_area = int(0.1*cell_px*cell_px)
    
    cells = wshed2d(norm_signal, level=4, min_area=min_area)
    
#    cell_image = Image.fromarray(cells, mode='I')
#    cell_image.save(output_dir+'cells.tif')

    cells = cells*mask2d
    cells = remove_small_large(cells, 0.1*cell_px*cell_px, 50*cell_px*cell_px)
    view_segmentation_lines(norm_signal, cells, fn=output_dir+'itk_segmentation.png')
    f.write('<h3> ITK Segmentation </h3>') 
    f.write('<p> {} cells found</p>\n'.format(np.max(cells)))

    f.write('<img src="itk_segmentation.png">\n') 

    if do_spm:
        cells = run_spm(norm_signal)
        cells = cells*mask2d
        cells = remove_small_large(cells, 0.1*cell_px*cell_px, 50*cell_px*cell_px)
        view_segmentation_lines(norm_signal, cells, fn=output_dir+'spm_segmentation.png')
        f.write('<h3> SPM Segmentation </h3>') 
        f.write('<p> {} cells found</p>\n'.format(np.max(cells)))
        f.write('<img src="spm_segmentation.png">\n') 
    

    areas, mu, mu0, _, slope = analyse_cells(cells, sps, spacing)

   # h = heatmap(cells, areas)

    vh = heatmap_array(cells, areas)

    plt.figure(figsize=(10,10))
    plt.imshow(vh, interpolation='none', cmap=plt.cm.viridis)
    plt.xticks()
    plt.yticks()
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.savefig(output_dir+'area_heatmap.png',  bbox_inches='tight')
    f.write('<h3> Area heatmap </h3>') 
    f.write('<img src="area_heatmap.png">\n') 


    vh = heatmap_array_shrunk (cells, areas)

    plt.figure(figsize=(10,10))
    plt.imshow(vh, interpolation='none', cmap=plt.cm.viridis)
    plt.xticks()
    plt.yticks()
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.savefig(output_dir+'area_heatmap_shrunk.png',  bbox_inches='tight')
    f.write('<h3> Area heatmap (shrunk_cells) </h3>') 
    f.write('<img src="area_heatmap_shrunk.png">\n') 


    vmu = heatmap_array(cells, mu)
    plt.figure(figsize=(10,10))
    plt.imshow(vmu, interpolation='none', cmap=plt.cm.viridis)
    plt.xticks()
    plt.yticks()
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.savefig(output_dir+'anisotropy_heatmap.png',  bbox_inches='tight')
    f.write('<h3> Anisotropy heatmap </h3>') 
    f.write('<img src="anisotropy_heatmap.png">\n') 

    vmu = heatmap_array_shrunk(cells, mu)
    plt.figure(figsize=(10,10))
    plt.imshow(vmu, interpolation='none', cmap=plt.cm.viridis)
    plt.xticks()
    plt.yticks()
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.savefig(output_dir+'anisotropy_heatmap_shrunk.png',  bbox_inches='tight')
    f.write('<h3> Anisotropy heatmap (shrunk cells) </h3>') 
    f.write('<img src="anisotropy_heatmap_shrunk.png">\n') 


    vmu0 = heatmap_array_shrunk(cells, mu0)
    plt.figure(figsize=(10,10))
    plt.imshow(vmu0, interpolation='none', cmap=plt.cm.viridis)
    plt.xticks()
    plt.yticks()
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.savefig(output_dir+'planar_anisotropy_heatmap_shrunk.png',  bbox_inches='tight')
    f.write('<h3> Planar anisotropy heatmap (shrunk cells) </h3>') 
    f.write('<img src="planar_anisotropy_heatmap_shrunk.png">\n') 



    vs = heatmap_array(cells, slope)
    plt.figure(figsize=(10,10))
    plt.imshow(vs, interpolation='none', cmap=plt.cm.viridis)
    plt.xticks()
    plt.yticks()
    plt.colorbar(fraction=0.03, pad=0.04)
    plt.axis('off')
    plt.savefig(output_dir+'slope_heatmap.png',  bbox_inches='tight')


    # Cell size histogram
    plt.figure()
    sns.distplot(areas.values())
    plt.xlabel('Area (microns^2)')
    plt.ylabel('Frequency')
    plt.savefig(output_dir+'cellsize_histogram.png')
    f.write('<h3> Cell size histogram </h3>') 
    f.write('<img src="cellsize_histogram.png">\n') 

    # Anisotropy histogram
    plt.figure()
    sns.distplot(mu.values())
    plt.xlabel('Anisotropy')
    plt.ylabel('Frequency')
    plt.savefig(output_dir+'anisotropy_histogram.png')
    f.write('<h3> Anisotropy histogram </h3>') 
    f.write('<img src="anisotropy_histogram.png">\n') 
    
    # Make Pandas dataframe

    f.write("<h3> Cell data table </h3>")
    df = pd.DataFrame({'area':areas, 'anisotropy':mu})

    print df
    df.to_html(f)

    df.to_csv(output_dir+'cell_data.csv')

    f.write("</body>\n</html>\n")

    f.close()

    plt.close('all')
#    plt.show()

if __name__=="__main__":
    import sys

    ma, spacing = load_lif_stack(sys.argv[1], int(sys.argv[2]))

    process_array(ma, spacing)
    stop_javabridge()
    
