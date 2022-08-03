#!/bin/bash

if [ -z "$1" ]; then
  echo "USAGE . ./make_data.sh splitnum"
else

SPLIT=$1

if [[ -f ./sepsisexp_timeseries_partition-A.tsv && \
      -f ./sepsisexp_timeseries_partition-B.tsv && \
      -f ./sepsisexp_timeseries_partition-C.tsv && \
      -f ./sepsisexp_timeseries_partition-D.tsv ]]; then
    echo "INFO: Data files exist, generating train/dev/test splits..."

    tmpfile=$(mktemp /tmp/abc-script.XXXXXX)

    if [ $SPLIT=0 ]; then
    
    # Split 0
    CNT=$(($(wc -l sepsisexp_timeseries_partition-A.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-A.tsv > "$tmpfile"
    CNT=$(($(wc -l sepsisexp_timeseries_partition-B.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-B.tsv >> "$tmpfile"
    sort -k1n "$tmpfile" > train.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-C.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-C.tsv > dev.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-D.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-D.tsv > test.dat

    elif [ $SPLIT=1 ]; then
    
    # Split 1
    CNT=$(($(wc -l sepsisexp_timeseries_partition-B.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-B.tsv > "$tmpfile"
    CNT=$(($(wc -l sepsisexp_timeseries_partition-C.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-C.tsv >> "$tmpfile"
    sort -k1n "$tmpfile" > train.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-D.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-D.tsv > dev.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-A.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-A.tsv > test.dat

    elif [ $SPLIT=1 ]; then
    
    # Split 2
    CNT=$(($(wc -l sepsisexp_timeseries_partition-C.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-C.tsv > "$tmpfile"
    CNT=$(($(wc -l sepsisexp_timeseries_partition-D.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-D.tsv >> "$tmpfile"
    sort -k1n "$tmpfile" > train.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-A.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-A.tsv > dev.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-B.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-B.tsv > test.dat

    elif [ $SPLIT=1 ]; then

    # Split 3
    CNT=$(($(wc -l sepsisexp_timeseries_partition-D.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-D.tsv > "$tmpfile"
    CNT=$(($(wc -l sepsisexp_timeseries_partition-A.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-A.tsv >> "$tmpfile"
    sort -k1n "$tmpfile" > train.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-B.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-B.tsv > dev.dat
   
    CNT=$(($(wc -l sepsisexp_timeseries_partition-C.tsv | awk '{print $1}')-1))
    tail -n $CNT sepsisexp_timeseries_partition-C.tsv > test.dat

    else 
        echo "ERROR: Split number out of range [0-3]."
    fi
    rm "$tmpfile"

    cut -f1 test.dat | uniq -c | sort -rn | sed 's/  */	/g' | sed 's/^	//'  > sorted_test_ids.dat
    cut -f1 dev.dat | uniq -c | sort -rn | sed 's/  */	/g' | sed 's/^	//' > sorted_dev_ids.dat
    cut -f1 train.dat | uniq -c | sort -rn | sed 's/  */	/g' | sed 's/^	//' > sorted_train_ids.dat

    . ./print_dev_ids.sh ${SPLIT}
    . ./print_test_ids.sh ${SPLIT}

    # Prepare data directory template
    mkdir -p data/data_empty
    ln -s train.dat data/data_empty/dev.dat
    ln -s train.dat data/data_empty/test.dat
    
    # Create other directories
    mkdir logs
    mkdir models

else
    echo "ERROR: Please extract the data archive 'SepsisExp.tar.gz' into the current directory."
fi

fi
