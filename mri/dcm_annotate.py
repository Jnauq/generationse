import os
import pydicom
import numpy as np
import cv2


#####################################################################################################

# For use with a single DICOM label file containing annotations for multiple different Dicom images.

#           Input:
#                   inputdir: input directory of dicom images
#                   annotationfile:  single dicom file containing labels
#
#           Output:
#                   a subdirectory of annotated images
#

# Note:
# Sometimes they provide labels and images in one directory.  In that case, image files often don't have
# .dcm extension (ie. IMG0001) and label files have .dcm extension (ie. IMG0001.dcm or IMG0001_p.dcm)

# May need to uncomment line 40 to filter out label file if it is located in same directory as images.
# Alternatively, move label to separate folder and leave line 40 commented out.

#####################################################################################################


inputdir = '/home/justin/Downloads/test1'                               # input directory of all image files
annotationfile = '/home/justin/Downloads/test1/label/test1.dcm'         # path for single Dicom label file


# Create dictionary of UID keys to filename values
def create_dict(indir):
    filedict = {}
    files = os.listdir(indir)
    files.sort()
    for file in files:
        if os.path.isfile(os.path.join(indir, file)):
            # if not file.endswith(".dcm") and not file.endswith(".txt"):     # Uncomment if .dcm labels exist in same directory as images
                ds = pydicom.dcmread(os.path.join(indir, file))
                if ds[0x8, 0x18].value not in filedict:                     # SOP Instance UID
                    filedict[ds[0x8, 0x18].value] = file
                else:
                    continue
    return filedict


# Read label dcm, make a list of Graphic Object Sequences.  True=Series, False=Single.
def read_annotations(annotations):
    ds = pydicom.dcmread(annotations)
    annlist = [i for i in ds[0x70, 0x1][0]]                                  # Graphic Annotation Sequence
    if (0x8, 0x1140) in [i.tag for i in annlist]:                            # Check for Referenced Image Sequence tag
        annlist = [i for i in ds[0x70, 0x1]]                                 # List of annotations for a Series of imgs
        return annlist, True
    else:
        annlist = [i for i in ds[0x70, 0x1][0][0x70, 0x9]]                   # List of annotations for a Single img
        return annlist, False


# Uses list of graphic data arrays to draw graphic
def draw_graphic(imgfile, datasets):
    save_dir = os.path.join(inputdir, "annotated")                           # image saved in subdir under input dir
    if not os.path.exists(save_dir):
        print('Making directory:', save_dir)
        os.mkdir(save_dir)

    ds = pydicom.dcmread(os.path.join(inputdir, imgfile))
    img = ds.pixel_array.astype(float)
    img = img / img.max() * 255.0
    img = np.uint8(img)

    # Need to convert image to RGB so annotation box is not grayscale, otherwise difficult to see
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                               # Convert grayscale to BGR

    for data in datasets:                                                     # Draw each set of data on image
        num_pts = len(data)

        if num_pts == 8:                                                      # Check if data type is ellipse
            data = np.array(data, dtype=np.int32)                             # cv2 ellipse only takes data as type int
            if data[0] == data[2]:
                cv2.ellipse(img, (data[0], data[7]), (data[6] - data[0], data[3] - data[7]), 0, 0, 360, (0, 0, 255), thickness=1)
            else:
                cv2.ellipse(img, (data[4], data[3]), (data[2] - data[4], data[7] - data[3]), 0, 0, 360, (0, 0, 255), thickness=1)

        else:
            pts = []
            for x in range(0, num_pts, 2):
                a = [data[x], data[x + 1]]
                pts.append(a)
            pts = np.array([pts], np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), thickness=1)

    cv2.imwrite(save_dir + '/{}.jpg'.format(os.path.basename(imgfile).split('.')[0]), img)     # (255-img) for inverse
    print('Created: {}.jpg'.format(imgfile.split('.')[0]))


# Creates list of all graphic data arrays for image
def annotate(annlist, series, filedict):
    if series:
        # Dicom with Annotations for a series of images
        for item in annlist:
            graphic_data = []
            graphic_objs = item[0x70, 0x9].VM                                # Num of Graphic Data per Graphic Sequence
            uid = item[0x8, 0x1140][0][0x8, 0x1155].value                    # The UID of image annotated
            filename = filedict[uid]                                         # Retrieve image associated with UID
            for i in range(0, graphic_objs):
                graphic_data.append(item[0x70, 0x9][i][0x70, 0x22].value)    # Each Graphic Data in Graphic Sequence
            draw_graphic(filename, graphic_data)
    else:
        # Dicom with Annotations for a single image
        graphic_data = []
        values = list(filedict.values())
        filename = values[0]
        for item in annlist:
            graphic_data.append(item[0x70, 0x22].value)                      # Each Graphic Data in a Graphic Sequence
        draw_graphic(filename, graphic_data)


fdict = create_dict(inputdir)
ann, isSeries = read_annotations(annotationfile)
annotate(ann, isSeries, fdict)
