#!/bin/bash

# create test predictions for ensembles (uniform+weighted)
#
for encnum in `cat test_ids.dat`

do

  echo "EncNum $encnum"

  mkdir -p "data/data_$encnum/"
  cp -a data/data_empty/* "data/data_$encnum/"

  grep "^$encnum" test.dat > "data/data_$encnum/train.dat"

  python ensemble_predictions.py $encnum > "data/data_$encnum/ensemble_test.dat"

done

