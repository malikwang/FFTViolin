#!/usr/bin/env python
# -*-coding:utf8-*-#
import os
import librosa
import copy
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
window = np.hamming(window_size)

def init_list():
	# 识别音符范围是G3~E7，MIDI Number从53到100，但左右各多取一个音符以便确定频率的截止带宽
	for midi in range(54,102):
		pitch = librosa.midi_to_note(midi)
		pitch_hz = librosa.note_to_hz(pitch)
		pitch_list.append(pitch)
		hz_list.append(pitch_hz)

def init_dic():
	# 这里的index是G3~E7在hz_list中的
	for index in range(1,47):
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

def sort_note(note):
	pitch_list = note.split('_')
	pitch_list.sort()
	return '_'.join(pitch_list)

def fft_transform(file_path,true_pitch):
	global correct_count
	x, sr = librosa.load(file_path,sr=None)
	x = librosa.resample(x, sr, target_sr)
	x_sample = x[10000:14096]
	y = fft(x_sample*window)
	abs_y = abs(y)
	# 获取每一个基频的能量
	energy_list = []
	for index in range(1,47):
		pitch = pitch_list[index]
		sample_list = pitch_sample_dic[pitch]
		pitch_energy = 0
		for sample_index in sample_list:
			pitch_energy += abs_y[sample_index]**2
		energy_list.append(pitch_energy)

	# 找第一个音
	tmp_energy_list = copy.deepcopy(energy_list)
	# 加上谐波的能量
	for index in range(len(energy_list)):
		try:
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+12])
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+19])
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+24])
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+28])
		except:
			pass
	pitch_index = np.argmax(energy_list)+1
	pred_pitch1 = pitch_list[pitch_index]
	try:
		octave_energy = energy_list[pitch_index-1-12]
		if 4*octave_energy > energy_list[pitch_index-1]:
			pred_pitch1 = pitch_list[pitch_index-12]
	except:
		pass

	# 找第二个音
	pitch_index = pitch_list.index(pred_pitch1)
	index = pitch_index - 1
	energy_list[index] *= 0.01
	try:
		energy_list[index+12] *= 0.01
		energy_list[index+19] *= 0.01
		energy_list[index+24] *= 0.01
		energy_list[index+28] *= 0.01
	except:
		pass

	tmp_energy_list = copy.deepcopy(energy_list)
	# 加上谐波的能量
	for index in range(len(energy_list)):
		try:
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+12])
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+19])
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+24])
			energy_list[index] += min(5*tmp_energy_list[index],tmp_energy_list[index+28])
		except:
			pass
	pitch_index = np.argmax(energy_list)+1
	pred_pitch2 = pitch_list[pitch_index]
	try:
		octave_energy = energy_list[pitch_index-1-12]
		if 4*octave_energy > energy_list[pitch_index-1]:
			pred_pitch2 = pitch_list[pitch_index-12]
	except:
		pass

	if pred_pitch1 == pred_pitch2:
		if pred_pitch1 == true_pitch:
			correct_count += 1
		else:
			print(pred_pitch2,true_pitch)
	else:
		pred_pitch = '%s_%s'%(pred_pitch1,pred_pitch2)
		if sort_note(pred_pitch) == sort_note(true_pitch):
			correct_count += 1
		else:
			print(pred_pitch,true_pitch)
		# index = pitch_list.index(true_pitch)-1
		# try:
		# 	print(tmp_energy_list[index],tmp_energy_list[index+12],tmp_energy_list[index+19],tmp_energy_list[index+24],tmp_energy_list[index+28])
		# 	index += 12
		# 	print(tmp_energy_list[index],tmp_energy_list[index+12],tmp_energy_list[index+19],tmp_energy_list[index+24],tmp_energy_list[index+28])
		# except:
		# 	pass

def cepstrum(file_path,true_pitch):
	global correct_count
	x, sr = librosa.load(file_path,sr=None)
	x = librosa.resample(x, sr, target_sr)
	ms_a = target_sr/2100
	ms_b = target_sr/100
	x_sample = x[10000:14096]
	window = np.hamming(window_size)
	y = fft(x_sample*window)
	C = fft(np.log(abs(y)));
	abs_C_sample = abs(C[ms_a:ms_b])
	max_index = np.argmax(abs_C_sample)
	fx = target_sr/(ms_a+max_index)
	pred_pitch = librosa.hz_to_note(fx)
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
	if '.' not in pitch_dir:
		pitch_dir_path = os.path.join(dataset_path, pitch_dir)
		for split_dir in os.listdir(pitch_dir_path):
			if '.' not in split_dir:
				split_dir_path = os.path.join(pitch_dir_path, split_dir)
				for wav_file in filter(os.listdir(split_dir_path)):
					wav_file_path = os.path.join(split_dir_path, wav_file)
					# if librosa.note_to_midi(pitch_dir) < librosa.note_to_midi('C4'):
					# 	cepstrum(wav_file_path,pitch_dir)
					# else:
					# 	fft_transform(wav_file_path,pitch_dir)
					fft_transform(wav_file_path,pitch_dir)
					total_count += 1

print(correct_count,total_count)