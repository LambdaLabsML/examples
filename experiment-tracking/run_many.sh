#!/bin/sh
for LR in 0.01 0.05
do
    for BS in 64 128
    do
        python train.py $@ --lr $LR --batch-size $BS
    done
done