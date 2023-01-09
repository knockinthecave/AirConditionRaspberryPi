import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('sensor.csv')

# csv헤더로 데이터 분류
datasets = df[['Temperature','Humidity','CO','CO2','O3','PM2.5']]
labels = df[['VOC']]

# 0 ~ 1로 정규화
scaler = MinMaxScaler()
scaler.fit(datasets)
scaled_datasets = scaler.transform(datasets)

#labels(Pandas -> Numpy) + ravel() 함수로 다차원 -> 1차원 배열로 변환
scaled_labels = np.ravel(labels.to_numpy())

"""
print(scaled_datasets)
print(numpy_labels)
"""

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(scaled_datasets, scaled_labels)

"""
x_test = [[0.3, 0.1, 0.2, 0.4, 0.4, 0.3]]
print(knn.predict(x_test))
"""

"""
23.01.09 ERROR :
Pandas -> Numpy 변환시 데이터셋은 2d array 즉, 2차원 배열로 변환되기 때문에 ravel()함수로 
1차원 배열로 만들면 안됨. ravel()함수 사용시 1차원배열로 바뀌어버려 오류 발생. 
즉, 정답지만 ravel함수로 1차원 배열로 변환해야한다.

input 들어올시 데이터 정규화 진행하도록 함수 구현하거나 해야할 필요 있음.

"""

