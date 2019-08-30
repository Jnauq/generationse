#!/usr/bin/env python

#######################################################################################
#  View metadata for a single Dicom file.

#  Usage: python metaimg.py <full path of file>
#  ie: python metaimg.py /home/justin/testRCH_feb/Data/cd21/SDY00003/SRS00002/IMG00000
#
#         python <full path of script>/metaimg.py IMG00000
#  ie.  python /home/justin/PycharmProjects/MRI/metaimg.py IMG00000
#######################################################################################

import pydicom
import sys


if len(sys.argv) < 2:
    print("Usage: metaimg.py <input_image>")
    sys.exit(1)

rootdir = sys.argv[1]

pydcm = pydicom.dcmread(rootdir)
print(pydcm)
