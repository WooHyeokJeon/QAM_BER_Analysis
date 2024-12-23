import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm #정규분포

mu=0 #평균
sigma = 1 #표준편차
num_sample = 1000000
s=np.random.normal(mu,sigma, num_sample) #평균 mu 표준편차 simga인 정규분포를 따르는 랜덤 변수 num_sanole개를 사용하시오

cnt, bins, ignored = plt.hist(s, 100, density=True)
plt.plot(bins, 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((bins-mu)**2)/(2*sigma**2)))
#Normal Dist.의 pdf를 그려본 것
plt.show()

#PDF
print(1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((1-mu)**2)/(2*sigma**2)))

#CDF
print("\n수학적 확률")
print(norm.cdf(2)) #수학적 확률

print("\n실제 뽑았을 때 확률")
print(np.sum(s<2)/num_sample) #실제 확률(값이 매번 달라짐)