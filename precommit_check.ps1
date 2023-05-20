# precommit_check.ps1

python main.py --preprocess PCA --n_bands 200 --model nn --dataset IndianPines --n_runs 100 --training_sample 0.1 > ./check/check_log