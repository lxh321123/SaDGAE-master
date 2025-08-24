#!/bin/bash
cd public/home/10201606469/single_cell2
echo "Current directory is $(pwd)"
# Declare an array for numbers
num=(11 46 8 5 2)

# Declare an array for names
names=("Young" "Chen" "Bach" "Quake_10x_Spleen" "Wang_Lung")


# Create or clear log file
#> log2.txt

# Loop over datasets, hvg, and neighbors
for i in {1..5..1}; do
    for j in {500..2000..500}; do
        count=0
        for k in {5..15..1}; do
            echo "dataset=${names[i-1]}, hvg=$j, neighbors=$k"
            if [[ $count -eq 0 ]]; then
                echo "pretrain AutoEncoder:"
                python ./ae/main.py --name="${names[i-1]}" --hvg="$j" --n_clusters="${num[i]}"
                echo "-------------------------------------------------------------------------------------------"
                count=1
            fi

            echo "pretrain GNN:"
            python ./GNN/main.py --name="${names[i-1]}" --hvg="$j" --neighbors="$k" --n_clusters="${num[i]}"
            echo "--------------------------------------------------------------------------------------------"

            # Logging results to log2.txt
            echo "dataset=${names[i-1]}, hvg=$j, neighbors=$k" >> log2.txt
            echo "pretrain:" >> log2.txt
            python ./pretrain/main.py --name="${names[i-1]}" --n_clusters="${num[i]}" --hvg="$j" --neighbors="$k" >> log2.txt
            echo "------------------------------------------------------------------------------------" >> log2.txt

            echo "formal train:"
            echo "dataset=${names[i-1]}, hvg=$j, neighbors=$k" >> log2.txt
            echo "formal train:" >> log2.txt
            python ./AGCN/main.py --name="${names[i-1]}" --n_clusters="${num[i]}" --hvg="$j" --neighbors="$k" >> log2.txt
            echo "completed! result: $?"
            echo "-------------------------------------------------------------------------------------"
            echo "-------------------------------------------------------------------------------------" >> log2.txt
        done
    done
done
