#!/bin/bash
# Corrida V6 (2026-09-26): ETL com a duracao real de cada ficheiro
# (Data Processing/time_duration_v6.csv). Mesmos protocolos da run_pr.sh.
# Saidas results_wesad_v6_*.csv, logs em logs_v6/.
# As corridas do WESAD usam a extraccao so do WESAD (etl_wesad): as linhas do
# WESAD saem pela mesma ordem que na extraccao completa, logo os splits sao os
# mesmos. A extraccao completa (etl_all) serve o combinado, o SWELL e a
# transferencia, e demora ~2 h por causa dos ficheiros do SWELL a 2048 Hz.
CSV=/home/eduarda/Desktop/ProcessedECGData_v6_wesad.csv
NPZ=/home/eduarda/Desktop/NNSeries_v6_wesad.npz
FULL=/home/eduarda/Desktop/ProcessedECGData_v6.csv
FULLNPZ=/home/eduarda/Desktop/NNSeries_v6.npz
DP="/home/eduarda/Desktop/repositorios/Thesis/Data Processing"
RAW="/home/eduarda/Desktop/Tese 2026/Datasets"
cd "/home/eduarda/Desktop/repositorios/Thesis/Classification Models"
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p logs_v6
case "$1" in
  etl_wesad)
    grep ECG_values "$DP/time_duration_v6.csv" > "$DP/time_duration_v6_wesad.csv"
    python3 -u "$DP/RPeakDetection_HRV_fixed.py" --datasets "$RAW" --durations "$DP/time_duration_v6_wesad.csv" \
      --out $CSV --series-out $NPZ --force > logs_v6/etl_wesad.log 2>&1 ;;
  etl_all)
    python3 -u "$DP/RPeakDetection_HRV_fixed.py" --datasets "$RAW" --durations "$DP/time_duration_v6.csv" \
      --out /home/eduarda/Desktop/ProcessedECGData_v6.csv --series-out /home/eduarda/Desktop/NNSeries_v6.npz \
      --force > logs_v6/etl_all.log 2>&1 ;;
  base)
    python3 wesad_baselines.py --csv $CSV --out results_wesad_v6_baselines.csv > logs_v6/baselines.log 2>&1 ;;
  abcd)
    for cv in random group loso; do
      python3 retrain_v5.py $CSV --models A,B,C,D --dataset WESAD --features F2 --seeds 5 --cv $cv \
        --out results_wesad_v6_${cv}_ABCD.csv > logs_v6/abcd_$cv.log 2>&1
    done ;;
  ef)
    for cv in random group loso; do for arch in summary seq; do
      python3 train_sequence.py --csv $CSV --series $NPZ --arch $arch --dataset WESAD --cv $cv --seeds 5 \
        --out results_wesad_v6_${cv}_${arch}.csv > logs_v6/${arch}_$cv.log 2>&1
    done; done ;;
  g)
    for cv in random group loso; do
      python3 train_sequence.py --csv $CSV --series $NPZ --arch both --dataset WESAD --cv $cv --seeds 5 \
        --out results_wesad_v6_${cv}_both.csv > logs_v6/both_$cv.log 2>&1
    done ;;
  pooled_abcd)
    for cv in random group; do
      python3 retrain_v5.py $FULL --models A,B,C,D --dataset all --features F2 --seeds 5 --cv $cv \
        --out results_v6_pooled_${cv}_ABCD.csv > logs_v6/pooled_abcd_$cv.log 2>&1
    done ;;
  pooled_efg)
    for cv in random group; do for arch in summary seq both; do
      python3 train_sequence.py --csv $FULL --series $FULLNPZ --arch $arch --dataset all --cv $cv --seeds 5 \
        --out results_v6_pooled_${cv}_${arch}.csv > logs_v6/pooled_${arch}_$cv.log 2>&1
    done; done ;;
  swell_transfer)
    for arch in summary seq both; do
      python3 train_sequence.py --csv $FULL --series $FULLNPZ --arch $arch --dataset SWELL --cv group --seeds 5 \
        --out results_v6_swell_group_${arch}.csv > logs_v6/swell_${arch}.log 2>&1
    done
    # 5 seeds (a V5 tinha 3), como no resto da tese
    python3 transfer_test.py --csv $FULL --series $FULLNPZ --seeds 5 \
      --out results_v6_transfer.csv > logs_v6/transfer.log 2>&1 ;;
  f1f2_xai)
    for f in F1 F2; do
      python3 retrain_v5.py $CSV --models D --dataset WESAD --features $f --seeds 5 --cv random \
        --out results_wesad_v6_${f}_D.csv > logs_v6/${f}_D.log 2>&1
    done
    python3 explain_model_g.py --csv $CSV --series $NPZ --weights saved_models/wesad_model_G_v6.weights.h5 \
      --out results/model_G_v6 > logs_v6/explain_g.log 2>&1
    python3 model_g_directions.py --csv $CSV --series $NPZ --weights saved_models/wesad_model_G_v6.weights.h5 \
      > logs_v6/directions_g.log 2>&1
    python3 model_g_permutation.py --csv $CSV --series $NPZ --weights saved_models/wesad_model_G_v6.weights.h5 \
      --out results_v6_model_g_permutation.csv > logs_v6/permutation_g.log 2>&1 ;;
esac
echo "FIM $1"
