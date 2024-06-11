import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# 모집단 생성
population1 = np.random.normal(loc=50, scale=10, size=100000)  # 모집단 1
population2 = np.random.normal(loc=60, scale=12, size=100000)  # 모집단 2

# 샘플링
sample_size1=81
sample_size2=101
sample1 = np.random.choice(population1, sample_size1, replace=True)  # 모집단 1에서 81개의 샘플 추출
sample2 = np.random.choice(population2, sample_size2, replace=True)  # 모집단 2에서 101개의 샘플 추출

# 샘플 분산 계산
var1 = np.var(sample1, ddof=1)  # 샘플 1의 분산
var2 = np.var(sample2, ddof=1)  # 샘플 2의 분산

# 분산 비율 계산
var_ratio = var1 / var2  # 샘플 1의 분산을 샘플 2의 분산으로 나눈 값

# F-분포를 이용한 신뢰구간 계산
alpha = 0.05
v1 = len(sample1) - 1  # 샘플 1의 자유도
v2 = len(sample2) - 1  # 샘플 2의 자유도

# F-분포의 임계값 계산
f_critical_lower = f.ppf(alpha / 2, v1, v2)
f_critical_upper = f.ppf(1 - (alpha / 2), v1, v2)

ci_lower = var_ratio * f_critical_lower  # 신뢰구간 하한
ci_upper = var_ratio / f_critical_lower  # 신뢰구간 상한

print(f"Variance Ratio: {var_ratio:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 신뢰구간 검증
num_iterations = 100000
count_within_ci = 0

for _ in range(num_iterations):
    sample1 = np.random.choice(population1, sample_size1, replace=True) # 복원추출
    sample2 = np.random.choice(population2, sample_size2, replace=True) # 복원추출
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    var_ratio = var1 / var2

    if ci_lower <= var_ratio <= ci_upper:
        count_within_ci += 1

# 신뢰구간 내에 있는 비율 계산
proportion_within_ci = count_within_ci / num_iterations
print(f"Proportion within 95% CI: {proportion_within_ci:.4f}")

# 결과 시각화
x = np.linspace(f.ppf(0.001, v1, v2), f.ppf(0.999, v1, v2), 1000)
y = f.pdf(x, v1, v2)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', lw=2, label='F-distribution')  # F-분포
plt.axvline(f_critical_lower, color='r', linestyle='--', label=f'{alpha/2*100:.1f} percentile (Critical Value)')  # 하한 임계값
plt.axvline(f_critical_upper, color='g', linestyle='--', label=f'{(1-alpha/2)*100:.1f} percentile (Critical Value)')  # 상한 임계값
plt.xlabel('Variance Ratio')  # x축 레이블
plt.ylabel('Density')  # y축 레이블
plt.title('F-distribution with Critical Values')  # 제목
plt.legend()  # 범례
plt.grid(True)  # 그리드 표시
plt.show()  # 그래프 출력