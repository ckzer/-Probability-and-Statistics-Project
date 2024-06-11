import numpy as np
import matplotlib.pyplot as plt

# 정규분포의 파라미터 설정
mu = 50     # 평균
sigma = 10  # 표준편차
population_size = 1500  # 모집단 크기

# 정규분포를 따르는 모집단 생성
population = np.random.normal(mu, sigma, population_size)

# 실제 데이터의 히스토그램 그리기

plt.figure(figsize=(12, 6))
count, bins, ignored = plt.hist(population, bins=30, density=True, alpha=0.6, color='b', edgecolor='black', label='Real Data')

# 확률 밀도 함수 (PDF) 그리기
xmin, xmax = plt.xlim()  # x축의 최소값과 최대값 설정
x = np.linspace(xmin, xmax, 100)  # PDF를 그릴 x값 설정
p = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)  # PDF 계산
plt.plot(x, p, 'k', linewidth=2, label='Probability Distribution Function')

# 제목과 라벨 추가
plt.title('Histogram of Real Data and Probability Distribution Function')  # 그래프 제목
plt.xlabel('Value')  # x축 라벨
plt.ylabel('Density')  # y축 라벨
plt.legend()  # 범례 표시

# 그래프 보여주기
plt.show()
