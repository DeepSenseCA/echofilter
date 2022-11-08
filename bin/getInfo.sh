#!/bin/bash

# For the mobile data, we had many variables that we wanted represented in each
# of the test/train/validate sets.  Instead of just using the filenames, we wanted
# to have some metadata with them.
#
# we ran ./getInfo.sh > files.txt
#
# used this as input to separate_mobile.sh

wdir=/data/dsforce/surveyExports/mobile/
infile="surveyInventory.csv"

echo "Filename, Tide, columns,Survey,Date Start,Date End,Season,Number of Grids,Complete (Y/N),Notes"

for survey in Survey* ; do

  cd $wdir
  inventory=`cat $infile | grep $survey`
  cd $survey

  for fname in *Sv_raw.csv ; do
    tide=""
    [[ $fname =~ "_F_" ]] && tide="Flow"
    [[ $fname =~ "_E_" ]] && tide="Ebb"

    width=`head -n 5 $fname | tail -n 1 | sed 's/[^,]//g' | wc -c `

    echo "${survey}/$fname , $tide , $width , $inventory"
  done
done
