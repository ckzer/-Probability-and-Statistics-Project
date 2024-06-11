import numpy as np
import matplotlib.pyplot as plt

# 모집단 설정
mu1 = 50   # 첫 번째 모집단의 평균
mu2 = 60   # 두 번째 모집단의 평균
sigma1 = 10  # 첫 번째 모집단의 표준 편차
sigma2 = 10  # 두 번째 모집단의 표준 편차
population_size = 1000000  # 모집단 크기

# 정규 분포를 따르는 두 개의 모집단 데이터 생성
population1 = np.random.normal(mu1, sigma1, population_size)
population2 = np.random.normal(mu2, sigma2, population_size)

# 샘플링
sample1 = np.random.choice(population1, 81)
sample2 = np.random.choice(population2, 101)

# 샘플의 평균과 표준 편차
mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
std1 = np.std(sample1, ddof=1)
std2 = np.std(sample2, ddof=1)

# 신뢰 구간 계산
alpha = 0.05  # 유의 수준
z = 1.96  # 신뢰 구간 95%에 해당하는 Z 값

# 모집단 분산을 알고 있는 경우
se_known = np.sqrt(sigma1**2 / 81 + sigma2**2 / 101)
ci_known = (mean1 - mean2) + np.array([-1, 1]) * z * se_known

# 모집단 분산을 모르지만 동일하다고 가정한 경우
pooled_var = ((80 * std1**2) + (100 * std2**2)) / (81 + 101 - 2)
se_pooled = np.sqrt(pooled_var * (1 / 81 + 1 / 101))
ci_pooled = (mean1 - mean2) + np.array([-1, 1]) * z * se_pooled

# 모집단 분산을 모르는 경우
se_unknown = np.sqrt(std1**2 / 81 + std2**2 / 101)
t = 1.984  # 자유도가 (81+101-2)인 t 분포의 95% 신뢰 구간에 해당하는 t 값
ci_unknown = (mean1 - mean2) + np.array([-1, 1]) * t * se_unknown

# 신뢰 구간 출력
print("Known Variance CI:", ci_known)
print("Same Variance CI:", ci_pooled)
print("Unknown Variance CI:", ci_unknown)

# 결과 시각화
plt.figure(figsize=(14, 10))

# 모집단 히스토그램
plt.subplot(2, 2, 1)
plt.hist(population1, bins=60, density=True, alpha=0.6, color='b', edgecolor='black', label='Population 1')
plt.hist(population2, bins=60, density=True, alpha=0.6, color='r', edgecolor='black', label='Population 2')
plt.title('Population Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# 샘플 히스토그램
plt.subplot(2, 2, 2)
plt.hist(sample1, bins=20, density=True, alpha=0.6, color='b', edgecolor='black', label='Sample 1')
plt.hist(sample2, bins=20, density=True, alpha=0.6, color='r', edgecolor='black', label='Sample 2')
plt.title('Sample Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# 신뢰 구간 그래프 - 가로선 그래프
plt.subplot(2, 1, 2)
plt.hlines(1, ci_known[0], ci_known[1], colors='blue')
plt.hlines(2, ci_pooled[0], ci_pooled[1], colors='orange')
plt.hlines(3, ci_unknown[0], ci_unknown[1], colors='green')

# 신뢰 구간 중심점 표시
plt.plot((mean1 - mean2), 1, 'bo')
plt.plot((mean1 - mean2), 2, 'ro')
plt.plot((mean1 - mean2), 3, 'go')

plt.yticks([1, 2, 3], ['Known Variance', 'Same Variance', 'Unknown Variance'])
plt.axvline(mean1 - mean2, color='gray', linestyle='--')
plt.title('95% Confidence Intervals for the Difference in Means')
plt.xlabel('Difference in Means')
plt.ylabel('Assumptions')

plt.tight_layout()
plt.show()
