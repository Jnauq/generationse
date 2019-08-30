import pydicom
import os
import cv2
import numpy as np

############################################################################################################
#  Using clahe to adjust brightness and contrast of images.  One label Dicom file per image Dicom file.
#
#           Input:
#               cl: clip limit
#               gs: grid size
#               save_dir: save directory
#               idir:  image directory
#               ldir:  label directory
#
#           Output:
#               directory of images augmented with clahe
#
############################################################################################################

cl = 3              # adjust clip limit
gs = 16             # adjust grid size

idir = '/home/justin/Downloads/learning_group2/decompressed'            # image directory
ldir = '/home/justin/Downloads/learning_group2/labels'                  # label directory
save_dir = '/home/justin/Downloads/learning_group2'                     # save directory

if not os.path.exists(save_dir):
    print('Making directory:', save_dir)
    os.mkdir(save_dir)

i_list = [os.path.join(idir, img) for img in os.listdir(idir)]
l_list = [os.path.join(ldir, label) for label in os.listdir(ldir)]

i_list.sort()
l_list.sort()

for x in range(len(i_list)):

    file1 = i_list[x]
    annfile = l_list[x]

    print(os.path.basename(file1))
    print(os.path.basename(annfile))

    ds = pydicom.dcmread(file1)
    img = ds.pixel_array.astype(float)
    print(img.shape)
    img = img / img.max() * 255.0
    img = np.uint8(img)

    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(gs, gs))
    cl_img = clahe.apply(img)

    cv2.imwrite(save_dir + '/clahe_{}_{}/{}.jpg'.format(str(cl), str(gs),
                os.path.basename(file1).split('.')[0]), cl_img)
    cv2.destroyAllWindows()








# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
# cl_img = clahe.apply(img)
# cl_img_i = clahe.apply(255-img)
#
# top = cv2.hconcat([img, cl_img])
# bottom = cv2.hconcat([255-img, cl_img_i])
#
# combined = cv2.vconcat([top, bottom])
#
# cv2.imwrite('./3_16_16.png', combined)
# cv2.imwrite('./clahe_{}_5_32_32.png'.format(os.path.basename(file1).split('.')[0]), combined)
