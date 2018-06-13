#!/usr/bin/env python
# -*-coding:utf8-*-#
import os
import librosa
import math
import numpy as np
from numpy.fft import fft,ifft

pitch_list = []
hz_list = []
pitch_sample_dic = {}
window_size = 4096
target_sr = 16000
total_count = 0
correct_count = 0

def init_list():
	for midi in range(54,97):
		pitch = librosa.midi_to_note(midi)
		pitch_hz = librosa.note_to_hz(pitch)
		pitch_list.append(pitch)
		hz_list.append(pitch_hz)

def init_dic():
	for index in range(1,42):
		low_band = (float(hz_list[index-1]) + float(hz_list[index])) / 2
		up_band = (float(hz_list[index]) + float(hz_list[index+1])) / 2
		low_index = int(low_band*window_size/target_sr)
		up_index = int(up_band*window_size/target_sr)
		pitch_sample_dic[pitch_list[index]] = range(low_index+1,up_index+1)

#滤除非音频文件
def filter(file_list):
	wav_list = []
	for file in file_list:
		if '.wav' in file:
			wav_list.append(file)
	return wav_list

def fft_transform(file_path,true_pitch):
	global correct_count
	x, sr = librosa.load(file_path,sr=None)
	x = librosa.resample(x, sr, target_sr)
	x_sample = x[10000:14096]
	window = np.hamming(window_size)
	y = fft(x_sample*window)
	abs_y = abs(y)
	# 获取每一个基频的能量
	energy_list = []
	for index in range(1,42):
		pitch = pitch_list[index]
		sample_list = pitch_sample_dic[pitch]
		pitch_energy = 0
		for sample_index in sample_list:
			pitch_energy += abs_y[sample_index]**2
		energy_list.append(pitch_energy)
	# 加上谐波的能量
	for index in range(len(energy_list)):
		try:
			energy_list[index] += energy_list[index+12]
			energy_list[index] += energy_list[index+19]
			energy_list[index] += energy_list[index+24]
			energy_list[index] += energy_list[index+28]
		except:
			pass
	pitch_index = np.argmax(energy_list)+1
	pred_pitch = pitch_list[pitch_index]
	# try:
	# 	octave_energy = energy_list[pitch_index-1-12]
	# 	if 2*octave_energy > energy_list[pitch_index-1]:
	# 		pred_pitch = pitch_list[pitch_index-12]
	# except:
	# 	pass
	if pred_pitch == true_pitch:
		correct_count += 1
	else:
		print(pred_pitch,true_pitch)

init_list()
# print(pitch_list)
# print(hz_list)
init_dic()
# print(pitch_sample_dic)

dataset_path = '/Users/lisimin/Desktop/Violin/Dataset/BUPT'
for pitch_dir in os.listdir(dataset_path):
	if '_' not in pitch_dir:
		pitch_dir_path = os.path.join(dataset_path, pitch_dir)
		for split_dir in os.listdir(pitch_dir_path):
			if '.' not in split_dir:
				split_dir_path = os.path.join(pitch_dir_path, split_dir)
				for wav_file in filter(os.listdir(split_dir_path)):
					wav_file_path = os.path.join(split_dir_path, wav_file)
					fft_transform(wav_file_path,pitch_dir)
					total_count += 1

print(correct_count,total_count)

# x, sr = librosa.load('/Users/lisimin/Desktop/B5_E5_4.wav',sr=None)
# ms_a = sr/2100
# ms_b = sr/100

# x_sample = x[20000:24096]
# window = np.hamming(len(x_sample))
# y = fft(x_sample*window)
# C = fft(np.log(abs(y)));
# abs_C_sample = abs(C[ms_a:ms_b])
# index_array = np.argsort(-abs_C_sample)
# for i in range(0,5):
# 	fx = sr/(ms_a+index_array[i]-1)
# 	print(librosa.hz_to_note(fx))