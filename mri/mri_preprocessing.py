import argparse
import os
from subprocess import call

#######################################################################################################
# Loop through parent directory and pre-processes all Nifti files (ANTs N4 and ANTS Rigid, Affine, SYN registration)
#  in their subdirectories.  Input directory is save directory set in dcm2niix.py  (outdirbase variable)

# Uncomment SYN block for non linear registration

# Usage: python MRI_preprocessing.py -i /home/justin/testRCH_feb/Nifti

########################################################################################################

parser = argparse.ArgumentParser(description='FSL BET skull stripping and ANTs Registration')
parser.add_argument('-i', '--input', action="store", help="input nifti directory")
args = parser.parse_args()

inputdir = os.path.dirname(args.input + "/")
outputdir = os.path.join(inputdir, "Preprocessed")

if not os.path.exists(outputdir):
    os.mkdir(outputdir)

filepathlist = []
for root, dirs, files in os.walk(inputdir):
    dirs.sort()
    files.sort()
    for file in files:
        if 'Preprocessed' not in root and file.endswith(".nii.gz"):
            filepathlist.append(os.path.join(root, file))

if (len(filepathlist) == 0):
    raise SystemExit("No Nifti files to pre-process in directory")

for inputfile in filepathlist:
    basename = os.path.basename(inputfile).split(".")[0]

    '''
    #print("BET call here")
    #ssfilename = outputdir + "/" + basename + "_ss.nii.gz"
    # -R robust, -A skull and scalp surfaces, -B reduce image bias
    #call(['bet', inputfile, ssfilename, "-R", "-A", "-B"])

    #if not os.path.exists(ssfilename):
        #print("FSL BET failed on: " + inputfile)
        #raise SystemExit("FSL BET failed on: " + inputfile)
    '''

    #print("N4BiasFieldCorrection call here")
    n4filename = outputdir + "/" + basename + "_n4.nii.gz"               # Output file for n4biasfieldcorrection
    n4bfc = "/usr/lib/ants/N4BiasFieldCorrection"
    n4call = [n4bfc, "-d", "3"]
    n4call += ["--input-image", inputfile]
    n4call += ["--bspline-fitting", "[50, 3]"]
    n4call += ["--shrink-factor", "4"]
    n4call += ["--convergence", "[50x50x50x50, 0.001]"]
    n4call += ["--histogram-sharpening", "[0.15, 0.01, 200]"]
    n4call += ["--output", n4filename]
    n4call += ["-v"]
    print(n4call)
    call(n4call)

    if not os.path.exists(n4filename):
        raise SystemExit("N4BiasFieldCorrection failed on: " + n4filename)

    regfilename = outputdir + "/" + basename + "_n4_RA.nii.gz"           # Output file for linear and nonlinear transformation
    outfile = "[out_," + regfilename + "]"
    t1brain = n4filename
    t1template = '/home/justin/MRI_Data/T1Template/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a.nii'

    if not os.path.exists(t1template):
        raise SystemExit("Template file does not exist: " + t1template)

    trans = "[" + t1template + "," + t1brain + "," + "1]"
    mi = "MI[" + t1template + "," + t1brain + ",1,32,Regular,0.25]"
    cc = "CC[" + t1template + "," + t1brain + ",1,4]"
    antreg = '/usr/lib/ants/antsRegistration'

    #print("ANTs call here.")
    cmdline = [antreg, "--dimensionality", "3"]
    cmdline += ["--float", "0"]
    cmdline += ["--output", outfile]
    cmdline += ["--interpolation", "BSpline"]
    cmdline += ["--winsorize-image-intensities", "[0.005, 0.997]"]
    cmdline += ["--use-histogram-matching", "0"]
    cmdline += ["--initial-moving-transform", trans]

    cmdline += ["--transform", "Rigid[0.1]"]
    cmdline += ["--metric", mi]
    cmdline += ["--convergence", "[100x80x50x10, 1e-8, 10]"]
    cmdline += ["--shrink-factors", "8x4x2x1"]
    cmdline += ["--smoothing-sigmas", "3x2x1x0vox"]

    cmdline += ["--transform", "Affine[0.1]"]
    cmdline += ["--metric", mi]
    cmdline += ["--convergence", "[100x80x50x10, 1e-8, 10]"]
    cmdline += ["--shrink-factors", "8x4x2x1"]
    cmdline += ["--smoothing-sigmas", "3x2x1x0vox"]

    # cmdline += ["--transform", "SyN[0.1,3,0]"]                            # SYN nonlinear transformation
    # cmdline += ["--metric", cc]
    # cmdline += ["--convergence", "[30x40x60x0, 1e-6, 10]"]
    # cmdline += ["--shrink-factors", "8x4x2x1"]
    # cmdline += ["--smoothing-sigmas", "1x0.5x0.2x0vox"]

    cmdline += ["-v"]  # verbose
    print(cmdline)
    call(cmdline)

    if not os.path.exists(regfilename):
        #print("ANTs registration failed on: " + n4filename)
        raise SystemExit("ANTs registration failed on: " + n4filename)
