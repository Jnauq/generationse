#!/usr/bin/env python

########################################################################################
# Create a CSV of the RCH data with filtered results.  Takes input of parent directory.

#  Input:
#       csv_name:   filename for csv file
#       modality:   type of scan CT or MRI
#       minslices:  minimum number of slices required
#       formats:    list of specific modalities to filter for
#
#   Output:
#       creates a .csv file containing metadata for dicom based on input filters

#  Usage: python metawalk.py  <parent directory contatining all dicom data>
#  ie: python metawalk.py /home/justin/testRCH_feb/Data

########################################################################################

import pydicom
import sys
import os
from pandas import DataFrame


if len(sys.argv) > 2:
    print("Usage: metawalk.py <input_directory>")
    sys.exit(1)

walkdir = sys.argv[1]
found_mod = False
index = 1
save_dir = sys.argv[1]                              # Output directory = Input directory
csv_name = 'usable_aug_rch_contents.csv'            # Filename for csv file

modality = 'MR'                                     # Either MR or CT
minslices = 99                                      # Minimum number of slices in volume
formats = ['mprage', 'MPRAGE', 'spgr', 'SPGR']      # Specific modalities to filter for


if not os.path.exists(save_dir):
    print('Making:', save_dir)
    os.mkdir(save_dir)

filepath = []       # full file path
fileid = []         # patient id
filesdy = []        # patient study
filesrs = []        # patient series
filestudy = []      # study descriptions
fileseries = []     # series descriptions
fileslices = []     # number of slices
filedate = []       # series date
filetime = []       # study time

for root, subdirs, files in os.walk(walkdir):
    subdirs.sort()
    files.sort()

    for file in files:
        if file.startswith('IMG'):
            imgcount = len(files)
            pydcm = pydicom.read_file(os.path.join(root, file))
            if modality in pydcm[0x8, 0x61].value and imgcount > minslices and any(name in pydcm[0x8, 0x103e].value for name in formats):        # filter out the various modalities and resolutions
                print(str(index) + '-----\n\t%s\n' % root)
                filepath.append(root)

                srs = root.split('/')[-1]
                filesrs.append(srs)
                sdy = root.split('/')[-2]
                filesdy.append(sdy)
                pid = root.split('/')[-3]
                fileid.append(pid)

                print('\t'+pydcm[0x8, 0x1030].value)
                filestudy.append(pydcm[0x8, 0x1030].value)

                print('\t'+pydcm[0x8, 0x103e].value)
                fileseries.append(pydcm[0x8, 0x103e].value)

                print('\t'+str(imgcount))
                fileslices.append(imgcount)

                print('\t'+pydcm[0x8, 0x21].value)
                filedate.append(pydcm[0x8, 0x21].value)

                print('\t'+pydcm[0x8, 0x31].value)
                filetime.append(pydcm[0x8, 0x31].value)

                found_mod = True
                index += 1
            break

if not found_mod:
    print('No %s found' % modality)


df = DataFrame({'File Path': filepath, 'CD/ID': fileid, 'SDY': filesdy, 'SRS': filesrs, 'Study Description': filestudy,
                'Series Description': fileseries, 'Number of Slices': fileslices,
                'Series Date': filedate, 'Series Time': filetime})
df.to_csv(save_dir + '/' + csv_name, sep=',', encoding='utf-8', index=False)    # Filepath and name for output csv
