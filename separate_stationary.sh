#!/bin/bash

# Take a list of dates, and separate them into test/train/validate sets
#
# One way to get that list would be something like:
# for survey in december2017  march2018  september2018 ; do ls -1 $survey/evExports/*.csv ; done | sed "s:-.*::g" >dates.txt

infile="dates.txt"
trainfile="train.txt"
traindates="date_train.txt"
testfile="test.txt"
testdates="date_test.txt"
valfile="validate.txt"
valdates="date_val.txt"
tempfile="temp.txt"
shuffile="shuffle.txt"

total=`cat $infile | wc -l`
testnum=`echo "scale=0;$total / 10" | bc  `
trainnum=`echo "scale=0;$total - ( 2 * $testnum )" | bc`
echo "Total : $total"
echo "Train : $trainnum"
echo "Test  : $testnum"

rm -rf $tempfile

# shuffle, and put enough in each file
shuf $infile > $shuffile
head -n $trainnum $shuffile > $traindates
tail -n `echo "scale=0;$testnum * 2" | bc` $shuffile > $tempfile
tail -n $testnum $tempfile > $valdates
head -n $testnum $tempfile > $testdates

rm -rf $tempfile

for line in `cat $traindates` ; do ls -1 ${line}*.csv ; done > $trainfile
for line in `cat $testdates` ; do ls -1 ${line}*.csv ; done > $testfile
for line in `cat $valdates` ; do ls -1 ${line}*.csv ; done > $valfile
