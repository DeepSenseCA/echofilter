#!/bin/bash

# Take a list of dates, and separate them into test/train/validate sets
#
# One way to get that list would be something like:
# for survey in december2017 march2018 september2018 ; do ls -1 $survey/*.csv ; done | sed "s:-.*::g" >dates.txt

wdir="/data/dsforce/surveyExports/MinasPassage/sets/temp"

infile="dates.txt"
trainfile="train.txt"
traindates="date_train.txt"
testfile="test.txt"
testdates="date_test.txt"
valfile="validate.txt"
valdates="date_val.txt"
tempfile="temp.txt"
shuffile="shuffle.txt"

myseed="1"
if [ $# -ge 1 ]; then
  myseed="$1"
fi

get_seeded_random() {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

total=`cat $infile | wc -l`
testnum=`echo "scale=0;$total / 10" | bc`
trainnum=`echo "scale=0;$total - ( 2 * $testnum )" | bc`
echo "Total : $total"
echo "Train : $trainnum"
echo "Test  : $testnum"

rm -rf "$tempfile"

# shuffle, and put enough in each file
shuf --random-source=<(get_seeded_random $myseed) "$infile" > "$shuffile"
head -n $trainnum $shuffile > "$traindates"
tail -n `echo "scale=0;$testnum * 2" | bc` "$shuffile" > "$tempfile"
tail -n $testnum "$tempfile" > "$valdates"
head -n $testnum "$tempfile" > "$testdates"

rm -rf "$tempfile"

for line in `cat "$wdir/$traindates"`; do ls -1 ${line}*.csv; done > "$wdir/$trainfile"
for line in `cat "$wdir/$testdates"`; do ls -1 ${line}*.csv; done > "$wdir/$testfile"
for line in `cat "$wdir/$valdates"`; do ls -1 ${line}*.csv; done > "$wdir/$valfile"
