#!/bin/bash

# Separate all of the survey files into test/train/validate sets

# Use the output from :
# /data/dsforce/surveyExports/getInfo.sh > files.txt

infile="allFiles.txt"
trainfile="train.txt"
testfile="test.txt"
valfile="validate.txt"
restfile="rest.txt"
tempfile="temp.txt"

# Check the number of data points for each file, count how many have that number of data points:
# cat $infile | awk -F, '{print $3}' | sort | uniq -c > columns.txt

# get the leaveout group
cat $infile | grep -e "_N2" -e "_S2" > leaveout.txt

# non-slick way of getting the compliment
rm -rf $tempfile
cat $infile | grep -v "_N2" > $tempfile
cat $tempfile | grep -v "_S2" > $restfile

# separate the winters, as they are a small subset
cat $restfile | grep Winter > winter.txt
cat $restfile | grep -v Winter > nonwinter.txt

total=`cat winter.txt | wc -l`
testnum=`echo "scale=0;$total / 10" | bc  `
trainnum=`echo "scale=0;$total - ( 2 * $testnum ) " | bc`

#echo "total: $total \n testnum: $testnum \n trainnum: $trainnum "
rm -rf $tempfile

# shuffle, and put enough in each file
shuf winter.txt > wintershuf.txt
head -n $trainnum wintershuf.txt > $trainfile
tail -n `echo "scale=0;$testnum * 2" | bc` wintershuf.txt > $tempfile
tail -n $testnum $tempfile > $valfile
head -n $testnum $tempfile > $testfile

# do the same, but for the "rest"
total=`cat nonwinter.txt | wc -l`
testnum=`echo "scale=0;$total / 10" | bc  `
trainnum=`echo "scale=0;$total - ( 2 * $testnum )" | bc`

rm -rf $tempfile

shuf nonwinter.txt > nonwintershuf.txt
head -n $trainnum wintershuf.txt > train.txt
tail -n `echo "scale=0;$testnum * 2" | bc ` wintershuf.txt > temp.txt
tail -n $testnum $tempfile > validate.txt
head -n $testnum $tempfile > test.txt

rm -rf $tempfile
