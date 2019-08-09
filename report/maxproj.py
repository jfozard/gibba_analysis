import sys
import numpy as np

from PIL import Image

from tifffile import TiffWriter, TiffFile, imsave



def load_tiff(fn):
    with TiffFile(fn) as tiff:
        data = tiff.asarray()#colormapped=False)
        print data.shape
        data = np.squeeze(data)
    return np.transpose(data, (1,2,0))

def maxproj(infile, outfile):
    im = load_tiff(infile)
    im = np.max(im, axis=2)

    print im.dtype, np.min(im), np.mean(im), np.max(im)

    if im.dtype==np.uint16:
        if np.max(im) < 4096:
            im = Image.fromarray((im/(4096/256)).astype(np.uint8))
        else:
            im = Image.fromarray((im/(256)).astype(np.uint8))
    else:
        im = Image.fromarray(im.astype(np.uint8))
    im.save(outfile)


if __name__ == '__main__':
    maxproj(sys.argv[1], sys.argv[2])
