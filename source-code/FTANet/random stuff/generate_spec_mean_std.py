import os
import sys
import glob
import pyprind
import numpy as np

# sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis\\FTANet_melodic')
# from utils.spec_feature_extraction import spec_extraction_wobatch

spec_list = []
for f in glob.glob("E:\Kuliah\S2\Thesis\source code thesis\database\spec_feature\*.npy"):
    spec_list.append(f)

# test concat
# feature1 = np.load(spec_list[0])
# feature2 = np.load(spec_list[1])
# fused = np.concatenate((feature1, feature2), axis=0)
# print(fused.shape)

# arrays = [np.array(x) for x in fused]
# mean_2007 = np.asarray([np.mean(k) for k in zip(*arrays)])
# std_2007 = np.asarray([np.std(k) for k in zip(*arrays)])
# print(mean_2007.shape, std_2007.shape)

loaded_feature = []
bar = pyprind.ProgBar(len(spec_list), track_time=True, title='Load Spectogram Feature')
for row in spec_list:
    feature = np.load(row)
    # print(feature.ndim)
    # if feature.ndim < 4:
    #     print(feature.ndim, row)
    loaded_feature.append(feature)
    bar.update()
# exit()
# print(np.asarray(loaded_feature).shape) # could not broadcast input array from shape (26,32,513,1) into shape (26,32,513)
# exit() all the input arrays must have same number of dimensions, but the array at index 0 has 4 dimension(s) and the array at index 2749 has 3 dimension(s)
print('Concat feature...')
fused = np.concatenate(loaded_feature, axis=0)
print(fused.shape)

arrays = [np.array(x) for x in fused]
mean = np.mean(arrays, axis=0)
std = np.std(arrays, axis=0)

print('Mean and Std feature...')
print(mean.shape, std.shape)
np.save('E:\\Kuliah\\S2\\Thesis\\source code thesis\\FTANet\\utils_s\\x_data_mean.npy', mean)
np.save('E:\\Kuliah\\S2\\Thesis\\source code thesis\\FTANet\\utils_s\\x_data_std.npy', std)