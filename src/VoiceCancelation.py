from __future__ import division

import sounddevice as sd

import numpy as np
import matplotlib.pyplot as plt
import math
import wave
from scipy.io import wavfile
from pylab import *
from scipy.io.wavfile import write


def find_gap(wave, diff_num, gap_weight):
    for i in range(len(wave) - diff_num):
        if abs(wave[i] + gap_weight) < abs(wave[i + diff_num]) :
            return i


def constant_wave_cancel_noise(wave, alfa=0.97):
    y = 0
    Y = []
    for i in range(len(wave)):
        y = (alfa * y) + ((1 - alfa) * wave[i])
        Y.append(y)
    return Y


def find_sub_freq(wave, time):
    imp_freq = []
    X = np.fft.fft(wave)
    N = x.size
    X_db = (20 * np.log10(2 * np.abs(X) / N))
    j = 0
    for i in range(int(math.ceil(len(time) / 2))):
        if X_db[i] >= -1:
            imp_freq.append(f[i])
            j = j + 1
    return imp_freq


def random_noise(wave, upper, lower, low_pure=0, up_pure=0):
    pure = np.linspace(low_pure, up_pure, len(wave))
    noise = np.random.normal(upper, lower, pure.shape)
    new_wave = wave + pure + noise
    return new_wave


def non_constant_wave_cancel_noise_1(wave):
    for i in range(len(wave)):
        wave[i] = wave[i] / 2
    for i in range(len((wave))):
        if (abs(wave[i]) < 3000):
            wave[i] = wave[i] * (0.03)
    wave = constant_wave_cancel_noise(wave, 0.96)
    for i in range(len(wave)):
        wave[i] = wave[i] * 2
    for i in range(len(wave)):
        wave[i] = wave[i] / 1.5
    for i in range(len((wave))):
        if (abs(wave[i]) < 2500):
            wave[i] = wave[i] * (0.07)
    wave = constant_wave_cancel_noise(wave, 0.97)
    for i in range(len(wave)):
        wave[i] = wave[i] * 1.5

# def non_constant_wave_cancel_noise_2(wave):
#     #find_gap(s1, 500, 1000)
#     new_wave = range(len(wave))
#     for i in range(len(wave)):
#         new_wave[i] = wave[i] / 2
#     for i in range(len((wave))):
#         if (abs(new_wave[i]) < 3000):
#             new_wave[i] = new_wave[i] * (0.03)
#     new_wave[i] = constant_wave_cancel_noise(new_wave, 0.96)
#     for i in range(len(wave)):
#         new_wave[i] = new_wave[i] * 2
#     for i in range(len(wave)):
#         new_wave[i] = new_wave[i] / 1.5
#     for i in range(len((wave))):
#         if (abs(new_wave[i]) < 2500):
#             new_wave[i] = new_wave[i] * (0.07)
#     new_wave[i] = constant_wave_cancel_noise(new_wave, 0.97)
#     for i in range(len(wave)):
#         new_wave[i] = new_wave[i] * 1.5
#     return np.array(new_wave)

sampFreq, snd = wavfile.read('/home/pooya/Downloads/homer_speech_rec_audio_detection_1459498101.wav', 'rw')
print(sampFreq)
print(snd)
print(snd.shape)
s1 = snd[:, 0]
print(s1)
timeArray = np.arange(0, 8863296, 1)
timeArray = timeArray / sampFreq
timeArray = timeArray * 1000  # scale to milliseconds

#find_gap(s1,500,1000)

non_constant_wave_cancel_noise_1(s1)

write('test1.wav', 44100, q1)

T = 10
Fs = 1000
t = np.arange(0, T * Fs) / Fs
x = np.sin(2 * np.pi * t) + np.sin(8 * np.pi * t)
N = x.size
f = (np.arange(0, N) * Fs / N)