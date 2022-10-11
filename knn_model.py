import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def train_knn_model(csv_file_path, neighbors):
    '''
    csv 파일의 경로와 이웃의 수를 받아 k-NN 모델을 학습해 모델을 반환한다
    
    csv_file_path: 불러올 csv 파일의 경로
    neighbors: k-NN 알고리즘의 이웃의 수

    csv 파일의 구성:
    0행 -> label 정보 (파일 이름, 피처 정보 등등...)
    1행~ -> 0열에는 파일 이름 (ex. 1.jpg,...) 1열에는 class
    '''
    raw_csv = pd.read_csv(csv_file_path)

    all_data = raw_csv.values[0:, 1:]
    x = all_data[:,1:].reshape(-1,99)
    y = all_data[:,0]

    knn_clasifier = KNeighborsClassifier(n_neighbors=neighbors)
    knn_clasifier.fit(x,y)

    return knn_clasifier

def predict_knn_model(knn_model, data):
    '''
    학습된 k-NN 모델로 데이터의 class를 예측해 결과를 반환한다
    
    knn_model: 학습된 k-NN 모델
    data: 예측하고자 하는 데이터
    
    데이터는 knn_model의 predict 함수로 전달될 때 2차원 배열로 변환된다
    '''
    return knn_model.predict(data.reshape(1,-1))