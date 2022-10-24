#!/bin/bash

dataset="mnist"

epochs=14
base=1431655765
test_kwargs='{"batch_size":1000,"sampler":true,"steps":1,"shuffle":false}'

for (( i=0; i<5; i++ ))
do
    seed=$((base-i))

    #--- PyTorch DataLoader
    time python mnist.py --dataset $dataset --train-kwargs '{"batch_size": 64, "sampler": false, "shuffle": true}' --epochs $epochs --seed $seed

    # --- Datasetq BaseLoader
    time python mnist.py --dataset $dataset --train-kwargs '{"batch_size": 64, "sampler": true, "steps": 1, "shuffle": true}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed

    # --- Datasetq DataqLoader
    time python mnist.py --dataset $dataset --train-kwargs '{"batch_size": 64, "sampler": true, "shuffle": true}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
    time python mnist.py --dataset $dataset --train-kwargs '{"batch_size": 64, "sampler": true, "shuffle": true, "max_visits": 10}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
    time python mnist.py --dataset $dataset --train-kwargs '{"batch_size": 64, "sampler": true, "shuffle": true, "max_visits": 25}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
    time python mnist.py --dataset $dataset --train-kwargs '{"batch_size": 64, "sampler": true, "shuffle": true, "max_visits": 50}' --test-kwargs $test_kwargs --epochs $epochs --seed $seed
done

# real	73m3.935s
# user	73m11.438s
# sys	0m35.263s

# real	62m18.002s
# user	94m23.972s
# sys	5m45.477s
