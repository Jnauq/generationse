import os
import pydicom

#####################################################################################
# Save all label metadata to txt file.
#
#       Input:
#           labeldir: directory containing labels
#           save_dir: save directory
#
#       Output:
#           directory of txt files containing label information from dicom metadata

######################################################################################


labeldir = '/home/justin/Downloads/learning_group/labels'       # input directory of label dicoms
save_dir = "/home/justin/Downloads/learning_group/txt"          # save directory

for i in os.listdir(labeldir):
    fs = i.split(" ")[0]
    print(fs)
    f = open(save_dir + "/{}.txt".format(fs), "w+")       # save directory and filename of text files

    ann = pydicom.dcmread(os.path.join(labeldir, i))
    num_objects = ann[0x70, 0x1][0][0x70, 0x9].VM

    for n in range(num_objects):
        print(ann[0x70, 0x1][0][0x70, 0x9][n][0x70, 0x22].value)
        f.write(str(ann[0x70, 0x1][0][0x70, 0x9][n][0x70, 0x22].value) + '\n')

    f.close()
