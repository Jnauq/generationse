import os
from subprocess import call
import pandas

#######################################################################################################
#  Converts all Dicoms to nifti for preprocessing

#       Input:
#           outdirbase:     save directory for nifti files
#           csv_filepath:   (Use .csv created from metawalk.py to create a list of filepaths of interest)
#
#       Output:
#           a directory tree of nifti files

#  Usage:  python dcmtonii.py

########################################################################################################

# requires dcm2niix
dcm2niix = '/usr/bin/dcm2niix'

outdirbase = '/home/justin/testRCH_aug/Nifti'                                   # Output directory for Niftis
csv_filepath = "/home/justin/testRCH_aug/Data/usable_aug_rch_contents.csv"      # Location of csv file


if not os.path.exists(outdirbase):
    print('Making:', outdirbase)
    os.mkdir(outdirbase)

df = pandas.read_csv(csv_filepath)
dflist = df['File Path'].values.tolist()            # Create list of filepaths of Dicoms to be converted


for id, filepath in enumerate(dflist):
    cd, sdy, srs = filepath.split("/")[-3],filepath.split("/")[-2],filepath.split("/")[-1]
    savename = cd + "_" + sdy + "_" + srs
    inputdir = dflist[id]
    outputdir = outdirbase + "/" + cd + "_nifti"
    # print(savename)
    # print(inputdir)
    # print(outputdir)

    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    call([dcm2niix, '-5',           # gz compression level (1=fastest..9=smallest, default 6)
          '-o', outputdir,          # output directory to match each id, study and series number of Dicom
          '-b', 'n',                # output bids in json file
          '-v', 'y',                # verbose
          '-z', 'y',                # output compressed
          '-m', 'y',                # merge slices from same series regardless of study time
          '-f', savename,           # filename for each saved nifti
          inputdir])                # loops through file path column in csv to feed full path as input
