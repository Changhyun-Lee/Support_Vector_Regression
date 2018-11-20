# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:16:34 2018

@author: chang
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from BA_SVR import svr_mj


# 평가 결과 확인
def test_result(y_pred, y_test):
    if y_pred.shape == y_test.shape :
        MAE = sum(abs(y_test-y_pred))/len(y_test)
        RMSE = np.sqrt(sum((y_test-y_pred)**2)/len(y_test))
        R2 = 1-sum((y_test-y_pred)**2)/sum((y_test-y_test.mean())**2)
    return np.array([round(MAE,4),round(RMSE,4), round(R2,4)])


# 샘플 데이터 함수 식
def test_function(x):
    return 3 + np.log(x) + np.sin(x)



# =====================================================================================================
# Sample Data 생성
# =====================================================================================================
n_samples = 50
np.random.seed(0)
x = np.random.rand(n_samples)*10
np.random.seed(1)
y = test_function(x) + np.random.randn(n_samples)/1.5
x = x.reshape(-1,1)

x_range = np.array(range(1,100)).reshape(-1,1)/10
y_actual = test_function(x_range)

plt.figure(figsize=(8,5))
plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
plt.plot(x,y,ls='',marker='o', c='b', alpha=0.5, label='sample data')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(loc=4, prop={'size': 11})
plt.title('Sample data plot')



# =====================================================================================================
# Linear regreesion
# =====================================================================================================
X_train, y_train = x.copy(), y.copy()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred1 = lr_model.predict(X_train)
y_pred3 = lr_model.predict(x_range)

plt.figure(figsize=(8,5))
plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.5)
plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(loc=4, prop={'size': 11})
plt.title('Linear Regression (A)')

result = np.zeros((1,3))
result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
print(result)

plt.figure(figsize=(8,5))
plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.5)
plt.plot(x_range, np.full(x_range.shape, y_train.mean()), ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(loc=4, prop={'size': 11})
plt.title('Linear Regression (B)')



# =====================================================================================================
# SV regreesion   - linear, c 변화 확인
# =====================================================================================================

c_list = [1,1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6]
result_sum = []
for C in c_list :
    epsilon = 0
    clf = SVR(C=C, epsilon=epsilon, kernel='linear')
    clf.fit(X_train, y_train)     
    
    y_pred1 = clf.predict(X_train)
    y_pred3 = clf.predict(x_range)
    
    plt.figure(figsize=(8,5))
    plt.fill_between(x_range.reshape(-1,), y_pred3.reshape(-1,)+epsilon, y_pred3.reshape(-1,)-epsilon, alpha=0.3)
    plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
    plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
    plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.5)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(loc=4, prop={'size': 12})
    plt.title('Support Vector Regression, C=%f'%C)
    
    result = np.zeros((1,3))
    result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
    print(result)
    
    result_sum = np.append(result_sum, C)
    result_sum = np.append(result_sum, clf.coef_)
    result_sum = np.append(result_sum, result)

result_sum = result_sum.reshape(-1,5)


# =====================================================================================================
# SV regreesion   - epsilon 변화 확인, C=0.1
# =====================================================================================================

C = 0.1
epsilon_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]
result_sum = []
for epsilon in epsilon_list :
    clf = SVR(C=C, epsilon=epsilon, kernel='linear')
    clf.fit(X_train, y_train)     
    
    y_pred1 = clf.predict(X_train)
    y_pred3 = clf.predict(x_range)
    
    plt.figure(figsize=(8,5))
    plt.fill_between(x_range.reshape(-1,), y_pred3.reshape(-1,)+epsilon, y_pred3.reshape(-1,)-epsilon, alpha=0.2, color='grey')
    plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
    plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
    plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.1)
    plt.plot(X_train[clf.support_],y_train[clf.support_],ls='',marker='o', c='b', alpha=0.5, label='Support Vector')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(loc=4, prop={'size': 12})
    plt.title('Support Vector Regression, epsilon=%f'%epsilon)
    
    result = np.zeros((1,3))
    result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
    print(result)
    
    result_sum = np.append(result_sum, epsilon)
    result_sum = np.append(result_sum, clf.coef_)
    result_sum = np.append(result_sum, result)

result_sum = result_sum.reshape(-1,5)


# =====================================================================================================
# margin, epsilong 그래프 만들기
# =====================================================================================================
x_range2 = np.array(range(1,100)).reshape(-1,1)/10-5

a=1
plt.figure(figsize=(8,5))
plt.fill_between(x_range2.reshape(-1,), (x_range2*a).reshape(-1,)+5, (x_range2*a).reshape(-1,)-5, alpha=0.2, color='grey')
plt.plot(x_range2, (x_range2*a), ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y = log(x) + sin(x)')
plt.xlim(-5,5)
plt.ylim(-10,10)
plt.title('Epsilon tube & margin')


# =====================================================================================================
# Kernel space
# =====================================================================================================

plt.figure(figsize=(8,5))
plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
plt.plot(x,y,ls='',marker='o', c='b', alpha=0.5, label='sample data')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(loc=4, prop={'size': 11})
plt.title('Original space')

lr_model = LinearRegression()
lr_model.fit(test_function(X_train), y_train)

y_pred1 = lr_model.predict(X_train)
y_pred3 = lr_model.predict(test_function(x_range))

plt.figure(figsize=(8,5))
plt.plot(test_function(x_range), y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
plt.plot(test_function(x_range), y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
plt.plot(test_function(x),y,ls='',marker='o', c='b', alpha=0.5, label='sample data')
plt.xlabel('3+log(x)+sin(x)', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(loc=4, prop={'size': 11})
plt.title('Feature space')



# =====================================================================================================
# SV regreesion   - rbf, c 변화 확인
# =====================================================================================================

c_list = [1000,100, 10,1,1E-1, 1E-2, 1E-3]
result_sum = []
for C in c_list :
    epsilon = 0
    clf = SVR(C=C, epsilon=epsilon, kernel='rbf')
    clf.fit(X_train, y_train)     
    
    y_pred1 = clf.predict(X_train)
    y_pred3 = clf.predict(x_range)
    
    plt.figure(figsize=(8,5))
    plt.fill_between(x_range.reshape(-1,), y_pred3.reshape(-1,)+epsilon, y_pred3.reshape(-1,)-epsilon, alpha=0.3)
    plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
    plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
    plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.5)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(loc=4, prop={'size': 12})
    plt.title('Support Vector Regression, C=%f'%C)
    
    result = np.zeros((1,3))
    result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
    print(result)
    
    result_sum = np.append(result_sum, C)
    result_sum = np.append(result_sum, result)

result_sum = result_sum.reshape(-1,4)



# =====================================================================================================
# SV regreesion   - epsilon 변화 확인, C=0.1, rbf
# =====================================================================================================

C = 1
epsilon_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]
result_sum = []
for epsilon in epsilon_list :
    clf = SVR(C=C, epsilon=epsilon, kernel='rbf')
    clf.fit(X_train, y_train)     
    
    y_pred1 = clf.predict(X_train)
    y_pred3 = clf.predict(x_range)
    
    plt.figure(figsize=(8,5))
    plt.fill_between(x_range.reshape(-1,), y_pred3.reshape(-1,)+epsilon, y_pred3.reshape(-1,)-epsilon, alpha=0.2, color='grey')
    plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y = log(x) + sin(x)')
    plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
    plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.1)
    plt.plot(X_train[clf.support_],y_train[clf.support_],ls='',marker='o', c='b', alpha=0.5, label='Support Vector')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(loc=4, prop={'size': 12})
    plt.title('Support Vector Regression, epsilon=%f'%epsilon)
    
    result = np.zeros((1,3))
    result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
    print(result)
    
    result_sum = np.append(result_sum, epsilon)
    result_sum = np.append(result_sum, result)

result_sum = result_sum.reshape(-1,4)




# =====================================================================================================
# SV regreesion   - gamma 변화 확인, C=1, rbf, ep=0.5
# =====================================================================================================

C = 1
epsilon = 0.5
gamma_list = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
result_sum = []
for gamma in gamma_list :
    clf = SVR(C=C, epsilon=epsilon, kernel='rbf', gamma=gamma)
    clf.fit(X_train, y_train)     
    
    y_pred1 = clf.predict(X_train)
    y_pred3 = clf.predict(x_range)
    
    plt.figure(figsize=(8,5))
    plt.fill_between(x_range.reshape(-1,), y_pred3.reshape(-1,)+epsilon, y_pred3.reshape(-1,)-epsilon, alpha=0.2, color='grey')
    plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y=3+log(x)+sin(x)')
    plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
    plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.5)
    plt.plot(X_train[clf.support_],y_train[clf.support_],ls='',marker='o', c='b', alpha=0.5, label='Support Vector')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend(loc=4, prop={'size': 12})
    plt.title('Support Vector Regression, gamma=%f'%gamma)
    result = np.zeros((1,3))
    result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
    print(result)
    
    result_sum = np.append(result_sum, gamma)
    result_sum = np.append(result_sum, result)

result_sum = result_sum.reshape(-1,4)




# =====================================================================================================
# define Loss
# =====================================================================================================
e=1
def e_insensitive(x, e):
    y = np.abs(x)-e
    idx = np.where(y<0)
    y[idx] = 0
    return y

def laplasian(x):
    y = np.abs(x)
    return y

def gaussian(x):
    y = 0.5*x**2
    return y

def huber(x, s):
    y = np.abs(x)
    idx = np.where(y <= s)
    y[idx] = 0.5/s*x[idx]**2
    idx = np.where(y > s)
    y[idx]=np.abs(x[idx])-0.5*s
    return y

def poly(x, p):
    y = (np.abs(x)**p)/p
    return y

def piece(x, s, p):
    y = np.abs(x)
    idx = np.where(y <= s)
    y[idx] = (np.abs(x[idx])**p)/p/(s**(p-1))
    idx = np.where(y > s)
    y[idx] = np.abs(x[idx])-s*(p-1)/p
    return y


x = np.arange(-3,3.1,0.1)

plt.figure(figsize=(8,5))
plt.plot(x,e_insensitive(x,1.5), label='e_insensitive(e=1.5)')
plt.plot(x,e_insensitive(x,1), label='e_insensitive(e=1.0)')
plt.plot(x,e_insensitive(x,0.5), label='e_insensitive(e=0.5)')
plt.ylim(-.2,3)
plt.title('loss function : e-intensitive')
plt.legend(loc=1, prop={'size': 12})

plt.figure(figsize=(8,5))
plt.plot(x,laplasian(x), label='laplasian')
plt.ylim(-.2,3)
plt.title('loss function : laplasian')
plt.legend(loc=1, prop={'size': 12})

plt.figure(figsize=(8,5))
plt.plot(x,gaussian(x), label='gaussian')
plt.ylim(-.2,3)
plt.title('loss function : gaussian')
plt.legend(loc=1, prop={'size': 12})

plt.figure(figsize=(8,5))
plt.plot(x,huber(x,3), label='huber(s=3)')
plt.plot(x,huber(x,2), label='huber(s=2)')
plt.plot(x,huber(x,1), label='huber(s=1)')
plt.plot(x,huber(x,0.00000001), label='huber(s~0)')
plt.ylim(-.2,3)
plt.title('loss function : hubers robust loss')
plt.legend(loc=1, prop={'size': 12})

plt.figure(figsize=(8,5))
plt.plot(x,poly(x,0.5), label='ploynomial(p=0.5)')
plt.plot(x,poly(x,1), label='ploynomial(p=1.0)')
plt.plot(x,poly(x,2), label='ploynomial(p=2.0)')
plt.plot(x,poly(x,3), label='ploynomial(p=3.0)')
plt.plot(x,poly(x,10), label='ploynomial(p=3.0)')
plt.ylim(-.2,3)
plt.title('loss function : polynomial')
plt.legend(loc=1, prop={'size': 12})


plt.figure(figsize=(8,5))
plt.plot(x,piece(x,1,0.5), label='piece(p=0.5)')
plt.plot(x,piece(x,1,1), label='piece(p=1.0)')
plt.plot(x,piece(x,1,2), label='piece(p=2.0)')
plt.plot(x,piece(x,1,5), label='piece(p=5.0)')
plt.ylim(-.2,3)
plt.title('loss function : piecewise polynomial')
plt.legend(loc=1, prop={'size': 12})


plt.figure(figsize=(8,5))
plt.plot(x,piece(x,0.5,2), label='piece(s=0.5)')
plt.plot(x,piece(x,1,2), label='piece(s=1.0)')
plt.plot(x,piece(x,2,2), label='piece(s=2.0)')
plt.plot(x,piece(x,5,2), label='piece(s=5.0)')
plt.ylim(-.2,3)
plt.title('loss function : piecewise polynomial')
plt.legend(loc=1, prop={'size': 12})

plt.plot(x,piece(x,1,5))




# =====================================================================================================
# loss 별 예측 성능 확인
# =====================================================================================================
t_size = 40
X_train, X_test, y_train, y_test = x[:t_size], x[t_size:], y[:t_size], y[t_size:]

epsilon = 0.5
C = 1
gamma = 1

loss_list = ['epsilon-insensitive', 'laplacian', 'gaussian', 'huber',  'polynomial', 'piecewise_polynomial', ]

loss = loss_list[5]
#svr = svr_mj(loss='epsilon-insensitive', kernel='linear', C=1.0, epsilon=epsilon, gamma=10)
#svr = svr_mj(loss='laplacian', kernel='linear', C=1, epsilon=epsilon, gamma=10)
svr = svr_mj(loss=loss, kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
svr.fit(X_train, y_train)     

y_pred1 = svr.predict(X_train)
y_pred3 = svr.predict(x_range)

idx1, idx2 = svr.get_idx()

plt.figure(figsize=(8,5))
plt.fill_between(x_range.reshape(-1,), y_pred3.reshape(-1,)+epsilon, y_pred3.reshape(-1,)-epsilon, alpha=0.2, color='grey')
plt.plot(x_range, y_actual, ls='-',marker='', color='grey', lw=3, alpha=0.5, label='y = log(x) + sin(x)')
plt.plot(x_range, y_pred3, ls='-',marker='', color='r', lw=3, alpha=0.5, label='y_pred')
plt.plot(X_train,y_train,ls='',marker='o', c='b', alpha=0.1)
plt.plot(X_train[idx1],y_train[idx1],ls='',marker='o', c='b', alpha=0.5, label='Support Vector')
plt.plot(X_train[idx2],y_train[idx2],ls='',marker='o', c='b', alpha=0.5, label='Support Vector')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(loc=4, prop={'size': 12})
plt.title('Support Vector Regression, loss=%s'%loss)

result = np.zeros((1,3))
result[0,0], result[0,1], result[0,2] = test_result(y_pred1, y_train)
print(result)













