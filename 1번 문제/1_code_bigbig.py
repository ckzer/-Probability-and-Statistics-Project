import numpy as np
import matplotlib.pyplot as plt

# 정규분포의 파라미터 설정
mu = 50     # 평균
sigma = 10  # 표준편차

# 모집단 크기 리스트
population_sizes = [1500000, 15000000]

fig, axes = plt.subplots(1, 2, figsize=(14, 10))
# 그래프 설정

# 각 모집단 크기에 대해 그래프 생성
for ax, population_size in zip(axes.flatten(), population_sizes):
    # 정규분포를 따르는 모집단 생성
    population = np.random.normal(mu, sigma, population_size)

    # 실제 데이터의 히스토그램 그리기
    count, bins, ignored = ax.hist(population, bins=80, density=True, alpha=0.6, color='b', edgecolor='black', label='Real Data')

    # 확률 밀도 함수 (PDF) 그리기
    xmin, xmax = ax.set_xlim()  # x축의 최소값과 최대값 설정
    x = np.linspace(xmin, xmax, 100)  # PDF를 그릴 x값 설정
    p = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)  # PDF 계산
    ax.plot(x, p, 'k', linewidth=2, label='Probability Distribution Function')

    # 제목과 라벨 추가
    ax.set_title(f'Population Size: {population_size}')  # 그래프 제목
    ax.set_xlabel('Value')  # x축 라벨
    ax.set_ylabel('Density')  # y축 라벨
    ax.legend()  # 범례 표시

# 레이아웃 조정 및 그래프 보여주기
plt.tight_layout()
plt.show()
