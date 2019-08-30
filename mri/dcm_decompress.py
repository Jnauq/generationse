from subprocess import call
import os


#############################################################################################################
# Input directory of dcms, a sub directory will be created within input dir containing all decompressed dcms

#       Input:
#               in_dir: directory location of dicom images
#
#       Output:
#               a sub directory, within input directory, containing decompressed dicoms

#############################################################################################################

#requires gdcm
gdcm = '/usr/bin/gdcmconv'

in_dir = '/home/justin/Downloads/test1'             # Fill in input directory
out_dir = in_dir + '/decompressed'                  # Output sub directory


if not os.path.exists(out_dir):
    print('Making directory:', out_dir)
    os.mkdir(out_dir)


def decompress(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.endswith('.dcm'):
            cmdline = [gdcm, '--raw']
            cmdline += [os.path.join(path, file), os.path.join(out_dir, file)]
            print(cmdline)
            call(cmdline)

decompress(in_dir)
