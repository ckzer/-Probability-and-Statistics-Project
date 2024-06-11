import numpy as np
import matplotlib.pyplot as plt

def generate_population(mu1, mu2, sigma1, sigma2, size):
    population1 = np.random.normal(mu1, sigma1, size)
    population2 = np.random.normal(mu2, sigma2, size)
    return population1, population2

def sample_population(population1, population2, n1, n2):
    sample1 = np.random.choice(population1, n1)
    sample2 = np.random.choice(population2, n2)
    return sample1, sample2

def calculate_confidence_intervals(sample1, sample2, sigma1=None, sigma2=None, alpha=0.05):
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    z = 1.96  # For 95% confidence interval
    
    # Known population variance
    if sigma1 is not None and sigma2 is not None:
        se_known = np.sqrt(sigma1**2 / len(sample1) + sigma2**2 / len(sample2))
        ci_known = (mean1 - mean2) + np.array([-1, 1]) * z * se_known
    else:
        ci_known = None

    # Unknown but equal population variance
    pooled_var = ((len(sample1) - 1) * std1**2 + (len(sample2) - 1) * std2**2) / (len(sample1) + len(sample2) - 2)
    se_pooled = np.sqrt(pooled_var * (1 / len(sample1) + 1 / len(sample2)))
    ci_pooled = (mean1 - mean2) + np.array([-1, 1]) * z * se_pooled

    # Unknown and unequal population variance
    se_unknown = np.sqrt(std1**2 / len(sample1) + std2**2 / len(sample2))
    t = 1.984  # For 95% confidence interval with degrees of freedom len(sample1) + len(sample2) - 2
    ci_unknown = (mean1 - mean2) + np.array([-1, 1]) * t * se_unknown

    return ci_known, ci_pooled, ci_unknown

def plot_confidence_intervals(ci_known, ci_pooled, ci_unknown, title):
    plt.figure(figsize=(14, 7))
    plt.hlines(1, ci_known[0], ci_known[1], colors='blue', label='Known Variance' if ci_known is not None else "")
    plt.hlines(2, ci_pooled[0], ci_pooled[1], colors='orange', label='Same Variance Assumed')
    plt.hlines(3, ci_unknown[0], ci_unknown[1], colors='green', label='Unknown Variance')

    mean_diff = (ci_pooled[0] + ci_pooled[1]) / 2
    plt.plot(mean_diff, 1, 'bo')
    plt.plot(mean_diff, 2, 'ro')
    plt.plot(mean_diff, 3, 'go')

    plt.yticks([1, 2, 3], ['Known Variance', 'Same Variance', 'Unknown Variance'])
    plt.axvline(mean_diff, color='gray', linestyle='--')
    plt.title(title)
    plt.xlabel('Difference in Means')
    plt.ylabel('Assumptions')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 모집단 설정
mu1, mu2 = 50, 60
sigma1, sigma2 = 10, 10
population_size = 1000000

# 평균 변화
mu1_new, mu2_new = 55, 65

# 분산 변화
sigma1_new, sigma2_new = 15, 15

# 인구 데이터 생성
population1, population2 = generate_population(mu1, mu2, sigma1, sigma2, population_size)
population1_new, population2_new = generate_population(mu1_new, mu2_new, sigma1, sigma2, population_size)
population1_var, population2_var = generate_population(mu1, mu2, sigma1_new, sigma2_new, population_size)

# 샘플링
sample1, sample2 = sample_population(population1, population2, 81, 101)
sample1_new, sample2_new = sample_population(population1_new, population2_new, 81, 101)
sample1_var, sample2_var = sample_population(population1_var, population2_var, 81, 101)

# 신뢰 구간 계산
ci_known, ci_pooled, ci_unknown = calculate_confidence_intervals(sample1, sample2, sigma1, sigma2)
ci_known_new, ci_pooled_new, ci_unknown_new = calculate_confidence_intervals(sample1_new, sample2_new, sigma1, sigma2)
ci_known_var, ci_pooled_var, ci_unknown_var = calculate_confidence_intervals(sample1_var, sample2_var, sigma1_new, sigma2_new)

# 신뢰 구간 출력
print("Original Known Variance CI:", ci_known)
print("Original Same Variance CI:", ci_pooled)
print("Original Unknown Variance CI:", ci_unknown)
print("Mean Change Known Variance CI:", ci_known_new)
print("Mean Change Same Variance CI:", ci_pooled_new)
print("Mean Change Unknown Variance CI:", ci_unknown_new)
print("Variance Change Known Variance CI:", ci_known_var)
print("Variance Change Same Variance CI:", ci_pooled_var)
print("Variance Change Unknown Variance CI:", ci_unknown_var)

# 결과 시각화
plot_confidence_intervals(ci_known, ci_pooled, ci_unknown, '95% Confidence Intervals (Original)')
plot_confidence_intervals(ci_known_new, ci_pooled_new, ci_unknown_new, '95% Confidence Intervals (Mean Change)')
plot_confidence_intervals(ci_known_var, ci_pooled_var, ci_unknown_var, '95% Confidence Intervals (Variance Change)')
