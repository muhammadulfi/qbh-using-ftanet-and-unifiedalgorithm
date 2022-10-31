import pickle

f = open('E:\Kuliah\S2\Thesis\source code thesis\database\precomputed_feature\ioacas_n3c_mnf.pkl',"rb")
feature = pickle.load(f)

for key, value in feature.items():
    print(key, value)