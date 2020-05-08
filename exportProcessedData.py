# python script to export binned stationary data

# Run this after the data has been cleaned/processed to export the binned data



# Change this to point to the directory where your survey is.
# it is assumed there will be a subdirectory called "EV Files", and will run on all .EV files in that directory
# There should already be a subdirectory called "evExports".  This will export to subdirectory of that, called binnedExports
# that is, export to "basedir/evExports/binnedExports/




#basedir='D:\\grandPassage20\\'
basedir='D:\\forceData\\wbat\\'


# Which echogram we're exporting from.  This probably won't change
varname = "Processed data 1"


# The reference line for the vertical bins.  This shouldn't change for stationary data
linename = "Nearfield"
# for mobile data, the reference should be bottom line:
#linename="Bottom line"



# exclude below this line.  For stationary, it should be "Nearfield".  For mobile, it should be the bottom line
aboveLine='Editable entrained air'
belowLine="Nearfield"
#aboveLine="Turbulence line"
#belowLine="Bottom line"


# The different types of binning to export
#  For each export, we need an array of 3 things
#  The first is the horizontal binning in minutes.  A really big number will give the whole transect, and something like 0.5 will be 30 seconds
#  The second is the vertical binning in meters.  Put in a large value like 200 for no vertical binning (ie, fullwater column)
#  The third is the suffix for the filename.  Normally a combination of vertical and horizontal bins in a readable form

# Feel free to remove one of these lines if that type of binning is no longer useful (it will speed up the process, if you don't export them all)
binSizes = [ \
            [0.5, 1, '30secTimeGrid_1mDepthBinsFromBottom'], \
            [0.5, 200, '30secTimeGrid_FullWaterColumn'], \
            [0.5, 0.5, '30secTimeGrid_halfmBinsFromBottom'], \
            [0.25, 1, '15secTimeGrid_1mDepthBinsFromBottom'], \
            [0.25, 200, '15secTimeGrid_FullWaterColumn'], \
            [0.25, 0.5, '15secTimeGrid_halfmBinsFromBottom'], \
            [0.1, 1, '6secTimeGrid_1mDepthBinsFromBottom'], \
            [0.1, 200, '6secTimeGrid_FullWaterColumn'], \
            [0.1, 0.5, '6secTimeGrid_halfmBinsFromBottom'], \

            ]


#            [1000000, 1, 'FullTransect_1mDepthBinsFromBottom'],\
#            [1, 1, '60secTimeGrid_1mDepthBinsFromBottom'], \
#            [1, 200, '60secTimeGrid_FullWaterColumn'], \
#            [1, 0.5, '60secTimeGrid_halfmBinsFromBottom'], \


# Before the template was changed, we'd have to make sure each of these options are checked so that they're all exported
# The echoview template should take care of this now
options=['Region_ID',  'Region_Name',  'Region_Class',  'Process_ID',  'Sv_Mean',  'NASC',  'Sv_Max',  'Sv_Min',  'Sv_Noise',  'NASC_Noise',  'Height_Mean',  'Depth_Mean',  'Good_Samples',  'PRC_NASC',  'Layer_Depth_Min',  'Layer_Depth_Max',  'Ping_S',  'Ping_E',  'Ping_M',  'Dist_S',  'Dist_E',  'Dist_M',  'VL_Start',  'VL_End',  'VL_Mid',  'Date_S',  'Time_S',  'Date_E',  'Time_E',  'Date_M',  'Time_M',  'Lat_S',  'Lon_S',  'Lat_E',  'Lon_E',  'Lat_M',  'Lon_M',  'Exclude_Below_Line_Depth_Mean',  'Program_Version',  'Processing_Version',  'Num_Layers',  'Num_Intervals',  'First_Layer_Depth_Start',  'Last_Layer_Depth_Stop',  'Processing_Date',  'Processing_Time',  'EV_Filename',  'Gain_Constant',  'Noise_Sv_1m',  'Minimum_Sv_Threshold_Applied',  'Minimum_Integration_Threshold',  'Maximum_Sv_Threshold_Applied',  'Maximum_Integration_Threshold',  'Exclude_Above_Line_Applied',  'Exclude_Above_Line_Depth_Mean',  'Exclude_Below_Line_Applied',  'Bottom_Offset',  'Uncorrected_Length',  'Uncorrected_Thickness',  'Uncorrected_Perimeter',  'Uncorrected_Area',  'Standard_Deviation',  'Attack_Angle',  'Corrected_Length',  'Corrected_Thickness',  'Corrected_Perimeter',  'Corrected_Area',  'Image_Compactness',  'Corrected_Mean_Amplitude',  'Corrected_MVBS',  'Coefficient_Of_Variation',  'Horizontal_Roughness_Coefficient',  'Vertical_Roughness_Coefficient',  '3D_School_Area',  '3D_School_Volume',  'ABC',  'ABC_Noise',  'PRC_ABC',  'Area_Backscatter_Strength',  'Num_Targets',  'TS_Mean',  'TS_Max',  'TS_Min',  'Target_Range_Mean',  'Target_Range_Max',  'Target_Range_Min',  'Speed_2D_Mean_Unsmoothed',  'Speed_4D_Mean_Unsmoothed',  'Speed_2D_Max_Unsmoothed',  'Speed_4D_Max_Unsmoothed',  'Direction_Horizontal',  'Direction_Vertical',  'Fish_Track_Change_In_Range',  'Time_In_Beam',  'Tortuosity_2D',  'Tortuosity_3D',  'Distance_2D_Unsmoothed',  'Distance_3D_Unsmoothed',  'Species_Id',  'Species_Name',  'Species_Percent',  'Species_TS',  'Species_Weight',  'Density_Number',  'Density_Weight',  'Bin_Count',  'Bin_TS_Mean',  'Thickness_Mean',  'Range_Mean',  'Layer_Range_Min',  'Layer_Range_Max',  'Exclude_Below_Line_Range_Mean',  'Exclude_Above_Line_Range_Mean',  'Target_Depth_Mean',  'Target_Depth_Max',  'Target_Depth_Min',  'Fish_Track_Change_In_Depth',  'First_Layer_Range_Start',  'Last_Layer_Range_Stop',  'Bad_Data_No_Data_Samples',  'Beam_Volume_Sum',  'No_Data_Samples',  'C_Good_Samples',  'C_Bad_Data_No_Data_Samples',  'C_No_Data_Samples',  'Wedge_Volume_Sampled',  'Region_Notes',  'Target_Length_Mean',  'Region_Detection_Settings',  'Grid_Reference_Line',  'Layer_Top_To_Reference_Line_Depth',  'Layer_Top_To_Reference_Line_Range',  'Layer_Bottom_To_Reference_Line_Depth',  'Layer_Bottom_To_Reference_Line_Range',  'Exclude_Below_Line_Depth_Min',  'Exclude_Below_Line_Range_Min',  'Exclude_Below_Line_Depth_Max',  'Exclude_Below_Line_Range_Max',  'Samples_Below_Bottom_Exclusion',  'Samples_Above_Surface_Exclusion',  'Samples_In_Domain',  'Bad_Data_Empty_Water_Samples',  'C_Bad_Data_Empty_Water_Samples',  'Bottom_Roughness_Normalized',  'Bottom_Hardness_Normalized',  'First_Bottom_Length_Normalized',  'Second_Bottom_Length_Normalized',  'Bottom_Rise_Time_Normalized',  'Heave_Source',  'Heave_Min',  'Heave_Max',  'Bottom_Line_Depth_Mean',  'Bottom_Max_Sv',  'Bottom_Kurtosis',  'Bottom_Skewness',  'Heave_Mean',  'Region_Bottom_Altitude_Min',  'Region_Bottom_Altitude_Max',  'Region_Bottom_Altitude_Mean',  'Region_Top_Altitude_Min',  'Region_Top_Altitude_Max',  'Region_Top_Altitude_Mean',  'Center_Of_Mass',  'Proportion_Occupied',  'Equivalent_Area',  'Aggregation_Index',  'Inertia',  'Kurtosis',  'Skewness',   'Alpha',  'Frequency']



# this first import is the one that allows the script to talk to the echoview application.  

import win32com.client
import os


# Open EV connection
evApp = win32com.client.Dispatch("EchoviewCom.EvApplication")
evApp.Minimize()





os.chdir(basedir)


evDir=os.path.join(basedir, 'EV_Files')
if not os.path.isdir(evDir):
    print('Error: cannot find evDir: ' + evDir)


#export directory
exDir=os.path.join(basedir, 'evExports')
if not os.path.isdir(exDir):
    print('Error: cannot find export dir: {}'.format(exDir))
binDir=os.path.join(exDir, 'binnedExports')
if not os.path.isdir(binDir):
    print('Error: cannot find export dir: {}'.format(binDir))

# find all EV files in this survey

evFileNames = [f for f in os.listdir(evDir) if f.endswith('.EV') ]



for evFileName in evFileNames:

    # open the EV file
    evFile = evApp.OpenFile(os.path.join( evDir, evFileName))
    print('  opened {}'.format(evFileName))

    # basename here is the EV file name, without the last 3 characters (i.e. without the ".EV" extension)
    basename=evFileName[11:-3])


    # Find the right variable
    av=evFile.Variables.FindByName(varname).AsVariableAcoustic()

    # Make sure we exclude the right regions (ie, export the cleaned/processed data) 
    av.Properties.Analysis.ExcludeAbove = aboveLine
    av.Properties.Analysis.ExcludeBelow = belowLine
    av.Properties.Analysis.ExcludeBadDataRegions = True


    #  again, we probably don't need to do this anymore, as the template should have all the export boxes checked
    for option in options:
        evFile.Properties.Export.Variables.Item(option).Enabled = True
    evFile.Save()

    for binSize in binSizes:		          

        # Use this as the reference line for depth grid
        refline = evFile.Lines.FindByName(linename)
        av.Properties.Grid.DepthRangeReferenceLine = refline

        # set the grid size
        # First number 0 = no depth grid ; 1 = use depth grid (ref is surface, depth=0) ; 2 = use ref line
        # second number = distance between grid lines
        av.Properties.Grid.SetDepthRangeGrid( 2, binSize[1] )
        
        # set the time grid size
        # first number
        #	0 = no time/distance grid; 
        #	1 = time in minutes; 
        #	2 = distance determined by GPS in nmi; 
        #	3 = distance according to vessel log in nmi; 
        # 	4 = distance in pings; 
        #	5 = distance using GPS in meters; 
        #	6 = distance using vessel log in meters
        # second number: 
        #	distance/time between grid lines (units depend on which number you put in first space)
        # Note: maximum time = 9999 minutes
        
        av.Properties.Grid.SetTimeDistanceGrid( 1, binSize[0] )


        
        fname= os.path.join(binDir, '{}_{}'.format(basename , binSize[2]))
        
        # export the settings file, and the actual binned data
        if not os.path.isfile(fname + '_settings.txt'):
            av.ExportIntegrationByCellsAll(fname +'.csv')
            print('    exported {}.csv'.format(fname))
          
            av.ExportSettings(fname + '_settings.txt')
        
                            
    evFile.Close()
                            
    print()

evApp.Quit()
print('Successfully completed')
    







