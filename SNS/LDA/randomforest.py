import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def ExportFeatureImport(model):
    n_feature = cancer.data.shape[1] #shape은 (인스턴스 개수, 속성 개수) 반환함
    plt.barh(np.arange(n_feature),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_feature),cancer.feature_names)
    plt.xlabel('Feature Import')
    plt.ylabel('Feature')
    plt.ylim(-1,n_feature)
    plt.show()


x,y = make_moons(n_samples=100, noise=0.25, random_state=3)
print(x,y)
x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y, random_state=42)

# print(x_train, x_test)
# print()
# print()
# print(y_train, y_test)

moon_forest = RandomForestClassifier(n_estimators=5, random_state=2)
moon_forest.fit(x_train, y_train)
print(type(x_train), type(y_train))
pre = moon_forest.predict(x_test)
print('vwrbwrb', pre)
print('forest사용 트리 5개 훈련 Data 모델적합도: {0:.3f}'.format(moon_forest.score(x_train, y_train)))
print('forest사용 트리 5개 시험 Data 모델적합도: {0:.3f}'.format(moon_forest.score(x_test, y_test)))

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target,
                                                    stratify=cancer.target, random_state=42)
forest100 = RandomForestClassifier(n_estimators=100, random_state=0)
forest100.fit(x_train, y_train)
print(type(x_train), type(y_train), np.shape(x_train), np.shape(y_train), type(x_train[0]), type(y_train[0]))
print(x_train, y_train)
print('forest사용 트리 100개 훈련 Data 모델적합도: {0:.3f}'.format(forest100.score(x_train, y_train)))
print('forest사용 트리 100개 시험 Data 모델적합도: {0:.3f}'.format(forest100.score(x_test, y_test)))

#ExportFeatureImport(forest100)

gbrt11 = GradientBoostingClassifier(random_state=0)
gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=3)
gbrt01 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt11.fit(x_train, y_train)
gbrt1.fit(x_train, y_train)
gbrt01.fit(x_train, y_train)

print('\n그레디어트부스팅 디폴트 훈련 Data 모델적합도: {0:.3f}'.format(gbrt11.score(x_train, y_train)))
print('그레디어트부스팅 디폴트 시험 Data 모델적합도: {0:.3f}'.format(gbrt11.score(x_test, y_test)))

print('\n그레디어트부스팅 깊이1 훈련 Data 모델적합도: {0:.3f}'.format(gbrt1.score(x_train, y_train)))
print('그레디어트부스팅 깊이1 시험 Data 모델적합도: {0:.3f}'.format(gbrt1.score(x_test, y_test)))

print('\n그레디어트부스팅 학습률0.1 훈련 Data 모델적합도: {0:.3f}'.format(gbrt01.score(x_train, y_train)))
print('그레디어트부스팅 학습률0.1 시험 Data 모델적합도: {0:.3f}'.format(gbrt01.score(x_test, y_test)))



