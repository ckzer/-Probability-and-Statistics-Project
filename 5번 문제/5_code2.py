import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# 단계 1: 두 모집단을 생성하고 샘플링
np.random.seed(42)
population1 = np.random.normal(loc=50, scale=10, size=100000)  # 모집단 1
population2 = np.random.normal(loc=60, scale=15, size=100000)  # 모집단 2

sample1 = np.random.choice(
    population1, 81, replace=False
)  # 모집단 1에서 81개의 샘플 추출
sample2 = np.random.choice(
    population2, 101, replace=False
)  # 모집단 2에서 101개의 샘플 추출

# 단계 2: 샘플 분산 계산
var1 = np.var(sample1, ddof=1)  # 샘플 1의 분산
var2 = np.var(sample2, ddof=1)  # 샘플 2의 분산

# 단계 3: 분산 비율 계산
var_ratio = var1 / var2  # 샘플 1의 분산을 샘플 2의 분산으로 나눈 값

# 단계 4: F-분포를 이용하여 신뢰구간 계산
alpha = 0.05
df1 = len(sample1) - 1  # 샘플 1의 자유도
df2 = len(sample2) - 1  # 샘플 2의 자유도

# F-분포의 임계값 계산
f_critical_lower = f.ppf(alpha / 2, df1, df2)
f_critical_upper = f.ppf(1 - alpha / 2, df1, df2)

ci_lower = var_ratio / f_critical_upper  # 신뢰구간 하한
ci_upper = var_ratio / f_critical_lower  # 신뢰구간 상한

print(f"Variance Ratio: {var_ratio:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 단계 5: 결과 시각화

# 그래프 1: F-분포와 임계값
x = np.linspace(f.ppf(0.001, df1, df2), f.ppf(0.999, df1, df2), 1000)
y = f.pdf(x, df1, df2)

plt.figure(figsize=(10, 6))
plt.plot(x, y, "b-", lw=2, label="F-distribution")  # F-분포
plt.axvline(
    f_critical_lower,
    color="r",
    linestyle="--",
    label=f"{alpha/2*100:.1f} percentile (Critical Value)",
)  # 하한 임계값
plt.axvline(
    f_critical_upper,
    color="g",
    linestyle="--",
    label=f"{(1-alpha/2)*100:.1f} percentile (Critical Value)",
)  # 상한 임계값
plt.xlabel("Variance Ratio")  # x축 레이블
plt.ylabel("Density")  # y축 레이블
plt.title("F-distribution with Critical Values")  # 제목
plt.legend()  # 범례
plt.grid(True)  # 그리드 표시
plt.show()  # 그래프 출력

# 그래프 2: 샘플 분산 비율과 신뢰구간
plt.figure(figsize=(10, 6))
plt.axvline(
    var_ratio, color="purple", linestyle="-", label=f"Variance Ratio (Sample)"
)  # 샘플 분산 비율
plt.axvspan(
    ci_lower, ci_upper, color="gray", alpha=0.2, label="95% CI for Variance Ratio"
)  # 신뢰구간 영역
plt.xlabel("Variance Ratio")  # x축 레이블
plt.ylabel("Density")  # y축 레이블
plt.title("Confidence Interval for Variance Ratio")  # 제목
plt.legend()  # 범례
plt.grid(True)  # 그리드 표시
plt.show()  # 그래프 출력
