import numpy as np
import matplotlib.pyplot as plt

# 첫 번째 모집단의 평균 및 두 번째 모집단의 평균
mu1 = 50   # 첫 번째 모집단의 평균
mu2 = 60   # 두 번째 모집단의 평균
sigma = 10  # 표준 편차 (동일한 분산 가정)
population_size = 1500  # population 크기

# 정규 분포를 따르는 두 개의 population 데이터 생성
population1 = np.random.normal(mu1, sigma, population_size)
population2 = np.random.normal(mu2, sigma, population_size)

# 히스토그램 및 Probablity Distribute Funcion를 그리기 위한 플롯 설정
plt.figure(figsize=(14, 7))

# 첫 번째 모집단의 히스토그램 플롯
plt.hist(population1, bins=60, density=True, alpha=0.6, color='b', label='Population 1 Real Data',edgecolor='black')

# 두 번째 모집단의 히스토그램 플롯
plt.hist(population2, bins=60, density=True, alpha=0.6, color='r', label='Population 2 Real Data',edgecolor='black')

# 첫 번째 모집단의 Probablity Distribute Funcion 계산 및 플롯
xmin1, xmax1 = plt.xlim()
x1 = np.linspace(xmin1, xmax1, 100)
p1 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1 - mu1) / sigma) ** 2)
plt.plot(x1, p1, 'b', linewidth=2, label='Population 1 Probablity Distribute Funcion')

# 두 번째 모집단의 Probablity Distribute Funcion 계산 및 플롯
x2 = np.linspace(xmin1, xmax1, 100)
p2 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2 - mu2) / sigma) ** 2)
plt.plot(x2, p2, 'r', linewidth=2, label='Population 2 Probablity Distribute Funcion')

# 그래프에 레이블과 제목 추가
plt.title(f'Population Size: {population_size}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# 그래프 표시
plt.grid(True)
plt.tight_layout()
plt.show()
