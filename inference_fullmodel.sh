#!/bin/bash

for encnum in `cat test_ids.dat`

do

  echo "EncNum $encnum"

  mkdir -p "data/data_$encnum/"
  cp -a data/data_empty/* "data/data_$encnum/"
  grep "^$encnum" train.dat > "data/data_$encnum/train.dat"

  python predict_ts.py --data "data/data_$encnum" \
                       --checkpoint "models/model_all_mixed.pt" \
                       --cuda --outf "data/data_$encnum/generated_all.dat"

  paste <(cut -f1-4 "data/data_$encnum/test.dat") \
        "data/data_$encnum/generated_all.dat" \
        > "data/data_$encnum/label_pred_all.dat"

done

