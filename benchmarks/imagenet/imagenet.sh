#!/bin/bash

dataset="imagenet"

epochs=90
base=1431655765
test_kwargs='{"batch_size":256,"sampler":true,"steps":1,"shuffle":false}'

for (( i=0; i<5; i++ ))
do
    seed=$((base-i))

    #--- PyTorch DataLoader
    # time python imagenet.py --dataset $dataset --train-kwargs '{"batch_size": 256, "sampler": false, "shuffle": true}' --epochs $epochs --seed $seed

    # --- Datasetq BaseLoader
    # time python imagenet.py --dataset $dataset --train-kwargs '{"batch_size": 256, "sampler": true, "steps": 1, "shuffle": true}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed

    # --- Datasetq DataqLoader
    # time python imagenet.py --dataset $dataset --train-kwargs '{"batch_size": 256, "sampler": true, "shuffle": true}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
    time python imagenet.py --dataset $dataset --train-kwargs '{"batch_size": 256, "sampler": true, "shuffle": true, "max_visits": 250}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
    # time python imagenet.py --dataset $dataset --train-kwargs '{"batch_size": 256, "sampler": true, "shuffle": true, "max_visits": 500}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
    # time python imagenet.py --dataset $dataset --train-kwargs '{"batch_size": 256, "sampler": true, "shuffle": true, "max_visits": 1000}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed

    break
done

# real	1915m32.759s
# user	11856m10.321s
# sys	755m44.036s
