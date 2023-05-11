#!/bin/bash
export DATA_PATH="../data/nle_data"
export SAVE_PATH="../data/nle_medium_data"

# All allowed role-race-align combos. See also: katakomba/utils/roles.py
combos=(
    "arc hum law"
    "arc hum neu"
    "arc dwa law"
    "arc gno neu"

    "bar hum neu"
    "bar hum cha"
    "bar orc cha"

    "cav hum law"
    "cav hum neu"
    "cav dwa law"
    "cav gno neu"

    "hea hum neu"
    "hea gno neu"

    "kni hum law"

    "mon hum neu"
    "mon hum law"
    "mon hum cha"

    "pri hum neu"
    "pri hum law"
    "pri hum cha"
    "pri elf cha"

    "ran hum neu"
    "ran hum cha"
    "ran elf cha"
    "ran gno neu"
    "ran orc cha"

    "rog hum cha"
    "rog orc cha"

    "sam hum law"

    "tou hum neu"

    "val hum neu"
    "val hum law"
    "val dwa law"

    "wiz hum neu"
    "wiz hum cha"
    "wiz elf cha"
    "wiz gno neu"
    "wiz orc cha"
)

for tup in "${combos[@]}"
do
    set -- $tup
    python3 generate_small_dataset.py \
        --data_path=$DATA_PATH \
        --save_path=$SAVE_PATH \
        --role="$1" --race="$2" --alignment="$3" \
        --num_episodes=700
done