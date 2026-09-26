#!/bin/bash
# Corrida de 2026-09-26: WESAD, 3 protocolos, com PR-AUC. Saidas results_wesad_pr_*.csv
CSV=/home/eduarda/Desktop/ProcessedECGData_v5.csv
NPZ=/home/eduarda/Desktop/NNSeries_v5.npz
cd "/home/eduarda/Desktop/repositorios/Thesis/Classification Models"
export TF_CPP_MIN_LOG_LEVEL=3
case "$1" in
  abcd)
    for cv in random group loso; do
      python3 retrain_v5.py $CSV --models A,B,C,D --dataset WESAD --features F2 --seeds 5 --cv $cv \
        --out results_wesad_pr_${cv}_ABCD.csv > logs_pr/abcd_$cv.log 2>&1
    done ;;
  ef)
    for cv in random group loso; do for arch in summary seq; do
      python3 train_sequence.py --csv $CSV --series $NPZ --arch $arch --dataset WESAD --cv $cv --seeds 5 \
        --out results_wesad_pr_${cv}_${arch}.csv > logs_pr/${arch}_$cv.log 2>&1
    done; done ;;
  g)
    for cv in random group loso; do
      python3 train_sequence.py --csv $CSV --series $NPZ --arch both --dataset WESAD --cv $cv --seeds 5 \
        --out results_wesad_pr_${cv}_both.csv > logs_pr/both_$cv.log 2>&1
    done ;;
esac
echo "FIM $1"
