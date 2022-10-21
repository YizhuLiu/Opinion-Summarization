## Synthetic training data creation
```
python train_condense.py -mode=train -data_type=yelp
```

## Basic aspect-guided model
```
  1 TOTAL_NUM_UPDATES=5000                                                                                                        
  2 WARMUP_UPDATES=200
  3 LR=3e-05
  4 MAX_TOKENS=800
  5 UPDATE_FREQ=32
  6 BART_PATH=/path/bart.large/model.pt
  7 
  8 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py data-bin \
  9     --restore-file $BART_PATH \
 10     --max-tokens $MAX_TOKENS \
 11     --task translation \
 12     --source-lang source --target-lang target \
 13     --truncate-source \
 14     --layernorm-embedding \
 15     --share-all-embeddings \
 16     --share-decoder-input-output-embed \
 17     --reset-optimizer --reset-dataloader --reset-meters \
 18     --arch bart_large \
 19     --criterion label_smoothed_cross_entropy \
 20     --label-smoothing 0.1 \
 21     --dropout 0.1 --attention-dropout 0.1 \
 22     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
 23     --clip-norm 0.1 \
 24     --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
 25     --update-freq $UPDATE_FREQ \
 26     --skip-invalid-size-inputs-valid-test \
 27     --find-unused-parameters \
 28     --ddp-backend=no_c10d \
 29     --required-batch-size-multiple 1 \
 30     --no-epoch-checkpoints \
 31     --save-dir checkpoints-basic \
 32     --seed 14632
```

## Advanced aspect-guided model
```
  1 TOTAL_NUM_UPDATES=5000 
  2 WARMUP_UPDATES=200      
  3 LR=3e-05
  4 MAX_TOKENS=350
  5 UPDATE_FREQ=32
  6 BART_PATH=/path/bart.large/model.pt
  7 
  8 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py data-bin-2 \
  9     --restore-file $BART_PATH \
 10     --max-tokens $MAX_TOKENS \
 11     --task translation \
 12     --source-lang source --target-lang target \                                                                               
 13     --truncate-source \
 14     --layernorm-embedding \
 15     --share-all-embeddings \
 16     --share-decoder-input-output-embed \
 17     --reset-optimizer --reset-dataloader --reset-meters \
 18     --arch bart_large \
 19     --criterion label_smoothed_cross_entropy \
 20     --label-smoothing 0.1 \
 21     --dropout 0.1 --attention-dropout 0.1 \
 22     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
 23     --clip-norm 0.1 \
 24     --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
 25     --update-freq $UPDATE_FREQ \
 26     --skip-invalid-size-inputs-valid-test \
 27     --find-unused-parameters \
 28     --ddp-backend=no_c10d \
 29     --required-batch-size-multiple 1 \
 30     --no-epoch-checkpoints \
 31     --patience 5 \
 32     --save-dir checkpoints-advance \
 33     --lr-weight 1000 \
 34     --T 0.2 \
 35     --multi-views \
 36     --balance \
                     
```

## Training advanced model initialzed with basic model
```
  1 TOTAL_NUM_UPDATES=5000 
  2 WARMUP_UPDATES=200      
  3 LR=3e-05
  4 MAX_TOKENS=350
  5 UPDATE_FREQ=32
  6 BART_PATH=/path/checkpoints-basic
  7 
  8 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py data-bin-2 \
  9     --restore-file $BART_PATH \
 10     --max-tokens $MAX_TOKENS \
 11     --task translation \
 12     --source-lang source --target-lang target \                                                                               
 13     --truncate-source \
 14     --layernorm-embedding \
 15     --share-all-embeddings \
 16     --share-decoder-input-output-embed \
 17     --reset-optimizer --reset-dataloader --reset-meters \
 18     --arch bart_large \
 19     --criterion label_smoothed_cross_entropy \
 20     --label-smoothing 0.1 \
 21     --dropout 0.1 --attention-dropout 0.1 \
 22     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
 23     --clip-norm 0.1 \
 24     --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
 25     --update-freq $UPDATE_FREQ \
 26     --skip-invalid-size-inputs-valid-test \
 27     --find-unused-parameters \
 28     --ddp-backend=no_c10d \
 29     --required-batch-size-multiple 1 \
 30     --no-epoch-checkpoints \
 31     --patience 5 \
 32     --save-dir checkpoints-advance-2 \
 33     --lr-weight 1000 \
 34     --T 0.2 \
 35     --multi-views \
 36     --balance \
```
