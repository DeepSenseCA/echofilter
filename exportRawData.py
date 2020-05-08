# Python script to export the requirements for Scott

# This will export the lines defined just below
# as well as the raw and cleaned data in csv files, and an image of the raw data


# Change this to point to the directory where your survey is.
# it is assumed there will be a subdirectory called "EV Files", and one called "evExports"
# This script will be run on all EV files in the first directory above, and export to the second
basedir = "D:\\grandPassage20\\"


# Variable Name:
#  This is the variable that we want to export the data from
#  We're not using the processed echogram, since we want the raw data
#  This probably won't change either
vname = "Fileset1: Sv pings T1"


# this first import is the one that allows the script to talk to the echoview application.

import win32com.client
import os


# Open EchoView connection
evApp = win32com.client.Dispatch("EchoviewCom.EvApplication")


os.chdir(basedir)


evDir = os.path.join(basedir, "EV Files")
if not os.path.isdir(evDir):
    print("Error: cannot find evDir: " + evDir)


# export directory
exDir = os.path.join(basedir, "evExports")
if not os.path.isdir(exDir):
    print("Error: cannot find export dir: {}".format(exDir))

# find all EV files in this survey

evFileNames = [f for f in os.listdir(evDir) if f.endswith(".EV")]


for evFileName in evFileNames:

    # open the EV file
    evFile = evApp.OpenFile(os.path.join(evDir, evFileName))

    print("  opened {}".format(evFileName))

    # basename here is the EV file name, without the last 3 characters (i.e. without the ".EV" extension)
    basename = evFileName[:-3]

    # Find the right variable
    av = evFile.Variables.FindByName(vname).AsVariableAcoustic()

    # make sure we don't exclude anything, ie export "raw" data
    av.Properties.Analysis.ExcludeAbove = "None"
    av.Properties.Analysis.ExcludeBelow = "None"
    av.Properties.Analysis.ExcludeBadDataRegions = False

    # Export the raw file:
    fname = os.path.join(exDir, basename + "_Sv_raw.csv")
    if not os.path.isfile(fname):
        av.ExportData(fname, -1, -1)
        print("    exported {}".format(fname))

    evFile.Close()

    print()

evApp.Quit()
print("Successfully completed")
