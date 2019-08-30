import pydicom
import os
import cv2
import numpy as np

###############################################################################################
#  For use with pairs of label dicoms and image dicoms.
#  Concatenates original image next to annotated image for comparison.

#          Input:
#               idir: directory of decompressed dicom images
#               ldir: directory of dicom labels
#               save_dir:  save directory
#
#          Output:
#               directory of annotated images

###############################################################################################


idir = '/home/justin/Downloads/learning_group2/decompressed'        # image directory
ldir = '/home/justin/Downloads/learning_group2/labels'              # label directory
save_dir = '/home/justin/Downloads/learning_group2/pngs'            # output directory

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

    print(file1)
    print(annfile)

    ds = pydicom.dcmread(file1)
    img = ds.pixel_array.astype(float)
    print(img.shape)
    img = img / img.max() * 255.0
    img = np.uint8(img)

    # Need to convert to RGB or else annotation box will be in grayscale
    img_og = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)      # Original image as RGB for side by side comparison
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)         # Image to be annotated as RGB


    ann = pydicom.dcmread(annfile)
    num_objects = ann[0x70, 0x1][0][0x70, 0x9].VM       # Number of annotation boxes for a dicom image

    for i in range(0, num_objects):
        list_points = ann[0x70, 0x1][0][0x70, 0x9][i][0x70, 0x22].value
        num_points = len(list_points)

        pts = []
        for x in range(0, num_points, 2):
            a = [list_points[x], list_points[x + 1]]
            pts.append(a)
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], True, (0, 69, 255), thickness=2)

    combined = cv2.hconcat([img, img_og])                # Two image arrays concatenated horizontally

    # Change save source from combined to img if you just want annotated image, or img_og if you want unannotated image
    cv2.imwrite(save_dir + '/{}.jpg'.format(os.path.basename(file1).split('.')[0]), combined)
    cv2.destroyAllWindows()



