# Import packages

import os

#from matplotlib.pyplot import plot, show, title

import scipy
import scipy.signal

import numpy
from numpy import loadtxt
from numpy import savetxt
import numpy as np

import biosppy
from biosppy.signals import ecg

from opensignalsreader import OpenSignalsReader

import pyhrv.tools as tools
from pyhrv.hrv import hrv

import csv
from csv import reader

from datetime import datetime

# Creating the final .csv file with all the information

hrv_csv_name = '/home/gsd/hrv_data/ProcessedECGData.csv'

with open(hrv_csv_name, 'a', encoding='UTF8', newline='') as f:
    header = ['file','label','hr_mean','nni_mean','nni_min','nni_max','sdnn','sdnn_index','rmssd','nn50','pnn50','LF/HF ratio','Total power']
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)


def split_ECG(time_frame, time, samples):
    
    batch = time_frame // time #number of samples that equal 30s

    number_of_arrays = samples // batch # number of total splits of ecg file

    split_values = np.array_split(filtered_ecg, number_of_arrays)
    
    return batch, number_of_arrays, split_values

def rPeakHamiltonTempo(filtered_ecg, fs, time):
    #Returns R-peak location indices (sample number, IT IS NOT TIME)
    rPeakLocation = biosppy.signals.ecg.hamilton_segmenter(signal=filtered_ecg, sampling_rate=fs)
    
    #Obtaining R-peak values in time (s): R-peak sample number -> R-peak time value
    tt = []
    
    for i in rPeakLocation:
        xx = i * time
        tt.append(xx)
        i = i + 1
        
    return tt



def rPeak(filtered_ecg, fs, time):
    #Returns R-peak location indices (sample number, IT IS NOT TIME)
    rPeakLocation = biosppy.signals.ecg.ecg(filtered_ecg, sampling_rate=fs, show = False)[2]
    
    #Obtaining R-peak values in time (s): R-peak sample number -> R-peak time value
    tt = []
    
    for i in rPeakLocation:
        xx = i * time
        tt.append(xx)
        i = i + 1
    
    return tt

#l is a list that contains x numbers of lists, x =number_of_arrays. 
#l[0] contains the time in which the rPeaks for the first 30s of measurmente occured.


def rPeaks_final(split_values, fs, time):
    rPeaks=[]
    for n in range (len(split_values)):
        #R-peak Hamilton
        #rPeakss = rPeakHamiltonTempo(split_values[n], fs, time)

        #R-peaks default parameters
        rPeakss = rPeak(split_values[n], fs, time)


        rPeaks.append(rPeakss)
    return rPeaks


def detectrr(rPeaks):
    
    #Obtaining RR intervals (difference in time between consecutive R-Peaks)
    rr = []
    for i in range(len(rPeaks) - 1):
        xx = rPeaks[i + 1] - rPeaks[i]
        rr.append(xx)
        i = i + 1
    return rr


#stops the function from printing the warnings
import warnings
warnings.filterwarnings('ignore')


##### Path
path = "/home/gsd/datasets/"



# open file in read 'r' mode
with open('/home/gsd/datasets/time_duration.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    
    start_time = datetime.now()
    for row in csv_reader:
        
        file = row[0]
        ecg_duration1 = row[1]
        ecg_duration = float(ecg_duration1)
        
        label1 = row[2]
        label = int(label1)
        
        fs1 = row[3]
        fs = int(fs1)
        
        # Join various path components
        final_path = os.path.join(path, file)
        
        
        print("Reading the file: ", file)
        
        filtered_ecg = loadtxt(final_path, delimiter=',')
        
        
        samples = sum(1 for row in filtered_ecg) # total number of data points in the ecg file
        time = ecg_duration / samples # time between consecutive data points in the .csv file
        time_frame = 30

        # Spliting the ECG in 30s
        batch, number_of_arrays, split_values = split_ECG(time_frame, time, samples)

        #Rpeaks for each 30s split
        rPeaks = rPeaks_final(split_values, fs, time)


        #HRV calculation and save .csv file
        x=len(split_values)


        #stops the hrv function from plotting all the graphics
        import matplotlib.pyplot as plt
        plt.ioff()

        for n in range (x):

            print("     Doing split ",n," of ",x)

            #Calculating RR interval
            rr_interval = detectrr(rPeaks[n])

            #Excluding falsed of missed R-peaks (in the case of rr_intervals < 0.5 or rr_intervals > 1.2 s)
            for g in rr_interval:
                if (g < 0.5 or g > 1.2):
                    rr_interval.remove(g)
                    
            #1. Calculating HRV based on NN (RR) intervals calculated before
            hrv1 = hrv(nni=rr_interval, sampling_rate=fs, show= False, plot_tachogram= False, plot_ecg= False)
            
            #2. Calculating HRV with filtered ecg, default parameters
            #hrv1 = hrv(signal=(split_values[n]), sampling_rate=fs,  show= False, plot_tachogram= False, plot_ecg= False)

            # the variable `hrv` is classbiosppy.utils.ReturnTuple(values, names=None), must be converted to ordered dictionary
            hrv_dic = hrv1.as_dict()
            hr_mean = hrv_dic["hr_mean"]
            nni_mean = hrv_dic["nni_mean"]
            nni_min = hrv_dic["nni_min"]
            nni_max = hrv_dic["nni_max"]
            sdnn = hrv_dic["sdnn"]
            sdnn_index = hrv_dic["sdnn_index"]
            rmssd = hrv_dic["rmssd"]
            nn50 = hrv_dic["nn50"]
            pnn50 = hrv_dic["pnn50"]
            X_ratio = hrv_dic["fft_ratio"]
            X_total= hrv_dic["fft_total"]

            data = [file, label, hr_mean, nni_mean,nni_min, nni_max, sdnn, sdnn_index, rmssd, nn50, pnn50, X_ratio, X_total]

            #open file in append 'a' mode
            with open(hrv_csv_name, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow(data)

            del hrv1 #cleaning the variable
            
        end_time = datetime.now()
        print('Elapsed Time: {}'.format(end_time - start_time))       
