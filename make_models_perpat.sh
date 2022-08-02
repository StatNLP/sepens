#!/bin/bash

# parallelize this loop
for i in `cut -f2 sorted_train_ids.dat`

do
  echo "EncNum $i"

  cp -a data/data_empty "data/data_$i"
  ln -sf ../../dev.dat "data/data_$i/dev.dat"
  
  grep "^$i" train.dat > "data/data_$i/train.dat"

  python main_ts.py --data "data/data_$i" --save "models/model_$i.pt" \
                    --cuda --epochs 20 --min_epochs 10 \
                    2>&1 | tee "logs/training_sepsis_$i.log"
  python predict_ts.py --data "data/data_$i" \
                       --checkpoint "models/model_$i.pt" \
                       --cuda --outf "data/data_$i/generated.dat"

  paste <(cut -f1-4 "data/data_$i/test.dat") "data/data_$i/generated.dat" \
        > "data/data_$i/label_pred.dat"
done
