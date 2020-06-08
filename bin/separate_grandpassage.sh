#!/bin/bash

# Separate GrandPassage dataset into test/train/validate sets

indir="/data/dsforce/surveyExports/GrandPassage"
outdir="/data/dsforce/surveyExports/GrandPassage/sets"

get_seeded_random() {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

myseed="1"
if [ $# -ge 1 ]; then
  myseed="$1"
fi

infile="files.txt"
trainfile="train.txt"
testfile="test.txt"
valfile="validate.txt"
tempfile="temp.txt"
shuffile="shuffle.txt"

for subdataset in "phase1" "phase2"; do

    echo "Partitioning $subdataset..."

    wdir="$outdir/$subdataset"
    mkdir -p "$wdir"

    cd "$indir"
    ls -1 ${subdataset}/*_Sv_raw.csv > "$wdir/$infile"
    cd "$wdir"

    total=`cat $infile | wc -l`
    testnum=`echo "scale=2;$total / 10" | bc`
    testnum=`printf "%.0f\n" $testnum`
    trainnum=`echo "scale=0;$total - ( 2 * $testnum )" | bc`
    echo "Total : $total"
    echo "Train : $trainnum"
    echo "Test  : $testnum"

    rm -rf "$tempfile"

    # shuffle, and put enough in each file
    shuf --random-source=<(get_seeded_random $myseed) "$infile" > "$shuffile"
    head -n $trainnum $shuffile > "$trainfile"
    tail -n `echo "scale=0;$testnum * 2" | bc` "$shuffile" > "$tempfile"
    tail -n $testnum "$tempfile" > "$valfile"
    head -n $testnum "$tempfile" > "$testfile"

    rm -rf "$tempfile"

done;

echo "Merging partitions for sub-datasets..."

# Merge the subdataset partitions together
mkdir -p "$outdir/firstpass"
for partition in "$trainfile" "$valfile" "$testfile"; do
    cat "$outdir/phase1/$partition" "$outdir/phase2/$partition" > "$outdir/firstpass/$partition"
done;

head -n 1 "$outdir/firstpass/$valfile" > "$outdir/firstpass/validate1.txt"
