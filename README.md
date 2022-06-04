# Deep Learning Approach for Stress Detection on ECG signals

This master thesis will focus on the detection of stress in real time, by the analysis of Electrocardiogram (ECG) signals, using Heart Rate Variability (HRV) data.
With this in mind, we used two datasets:

1. The Swell Knowledge work dataset for Stress and User modeling research [(SWELL-KW)](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624)
2. WESAD: Multimodal Dataset for Wearable Stress and Affect Detection [(WESAD)](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

Following the ETL process we first extracted data from the datasets so it could be later analysed. The process occurs as follows: first a band-pass filter was applied to the signal (0.5 Hz and 40 Hz), and then the QRS complex detection stage occurs. Subsequently, it was necessary to split the signal into 30 seconds segments, to recreate a real-time stress detection scenario. RR-interval series are estimated for each segment with the purpose of doing feature extraction. False or missed R-peaks were eliminated, by removing the RR-intervals that were outside the reach of 500 to 1200 ms.
