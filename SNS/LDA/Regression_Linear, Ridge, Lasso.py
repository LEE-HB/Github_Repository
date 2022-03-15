import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

'''
<선형회귀 -> Ridge(릿지) 분석(L2 규제) -> Lasso(라쏘)>
- 선형회귀 대안으로 릿지가 나왔으며 L2 규제임
- 릿지 대안으로 라소가 나왔고 L1 규제임
   => 릿지와 라소 둘 다 계수를 0으로 만들려고 하지만 다른 점은 라소는 정말 0으로 만들어 특정 속성을 사용하지 않고 
   릿지는 0에 가깝게 만드는 것 뿐임 즉 특정속성을 제거하지 않음
   => 라쏘는 특성 선택(Feture select)를 자동으로 해줌
   ?? L1, L2 규제가 무엇인지 알아보자
- 릿지와 라소 회귀분석은 alpha값을 이용하여 과소과대적합에 대한 정도를 조종 할 수 있음
   => alpha값이 커질 수록 계수를 0에 가깝게 만듦
  *주의*
   alpha 값이 0이면 선형회귀랑 같은 결과를 나타냄
   alpha 값이 증가하면 과소적합 감소하면 과대적합이됨, 선형회귀는 자유로운 모델이어서 과대적합이 될 확률이 높지만 릿지와 라소는 
   덜 자유롭기 때문에 과대적합이 적어지게 된다.
  
'''
'''
# -아래는 선형회귀 릿지 라소회귀 분석을 알아보는 코딩임
#x,y = mglearn.datasets.make_wave(n_samples=60)
x,y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

lr = LinearRegression().fit(x_train, y_train)
print(lr)

print("훈련 R^2:{0:.2f}".format(lr.score(x_train,y_train)))
print("시험 R^2:{:.2f}".format(lr.score(x_test,y_test)))


ridge1 = Ridge(alpha=1).fit(x_train,y_train)
ridge01 = Ridge(alpha=0.1).fit(x_train,y_train)
ridge10 = Ridge(alpha=10).fit(x_train,y_train)
ridge0 = Ridge(alpha=0).fit(x_train,y_train)
#print(ridge1.coef_) #전체 특성의 계수를 전부 뽑아줌

lasso1 = Lasso(alpha=1).fit(x_train, y_train)
lasso01 = Lasso(alpha=0.01, max_iter=100000).fit(x_train, y_train) # max_iter을 적용하지 않으면 경고 발생
lasso001 = Lasso(alpha=0.0001, max_iter=1000000).fit(x_train, y_train)
#print(lasso.coef_)
print(np.sum(lasso1.coef_ != 0))
print(np.sum(lasso01.coef_ != 0))
print(np.sum(lasso001.coef_ != 0))

# ============== 회귀분석 & 릿지분석 ==========================
# plt.plot(ridge10.coef_,'^',label='Ridge alpha=10')
# plt.plot(ridge1.coef_,'s',label='Ridge alpha=1')
# plt.plot(ridge0.coef_,'o',label='Ridge alpha=1')
# plt.plot(ridge01.coef_,'v',label='Ridge alpha=01')
# #plt.plot(lr.coef_, 'o', label = 'LinearRegression')
# plt.xlabel('계수 목록')
# plt.ylabel('계수 크기 ')
# x_lim = plt.xlim()
# plt.hlines(0,x_lim[0],x_lim[1])
# plt.xlim(x_lim)
# plt.ylim(-25,25)
# plt.legend()
# plt.show()

# ============== 라소 & 릿지분석 ==========================
# plt.plot(lasso1.coef_,'s',label='lasso alpha=1')
# plt.plot(lasso01.coef_,'^',label='lasso alpha=01')
# plt.plot(lasso001.coef_,'v',label='lasso alpha=001')
# plt.plot(ridge01.coef_,'o',label='Ridge alpha=01')
# plt.plot(lr.coef_, 'o', label = 'LinearRegression')
# plt.xlabel('계수 목록')
# plt.ylabel('계수 크기 ')
# x_lim = plt.xlim()
# plt.hlines(0,x_lim[0],x_lim[1])
# plt.xlim(x_lim)
# plt.ylim(-25,25)
# plt.legend()
# plt.show()
'''