import numpy as np
import matplotlib.pyplot as plt

# 정규분포의 파라미터 설정
mu = 50     # 평균
sigma = 10  # 표준편차
population_size = 1000000

# 정규분포를 따르는 모집단 생성
population = np.random.normal(mu, sigma, population_size)

# 샘플 크기 설정
sample_sizes = [10, 30, 61]

# 샘플링 반복 횟수 설정
num_repetitions = 5

# 샘플 평균과 샘플 분산을 계산하는 함수
def calculate_sample_statistics(sample):
    sample_mean = np.mean(sample)  # 샘플 평균 계산
    sample_variance = np.var(sample, ddof=1)  # 샘플 분산 계산 (n-1 사용)
    return sample_mean, sample_variance

# 결과 저장소
results = {}

# 각 샘플 크기에 대해 샘플링 반복
for sample_size in sample_sizes:
    sample_means = []
    sample_variances = []
    
    for _ in range(num_repetitions):
        # 모집단에서 무작위로 샘플 추출 (중복 허용 안함)
        sample = np.random.choice(population, sample_size, replace=False)
        sample_mean, sample_variance = calculate_sample_statistics(sample)
        sample_means.append(sample_mean)  # 샘플 평균 저장
        sample_variances.append(sample_variance)  # 샘플 분산 저장
    
    # 결과 저장
    results[sample_size] = {
        'Sample Means': sample_means,
        'Sample Variances': sample_variances
    }

# 모집단 분포를 시각화
plt.figure(figsize=(10, 6))
plt.hist(population, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Population Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# 샘플 분포를 시각화
fig, axes = plt.subplots(3, num_repetitions, figsize=(20, 12))

for i, sample_size in enumerate(sample_sizes):
    for j in range(num_repetitions):
        sample = np.random.choice(population, sample_size, replace=False)
        sample_mean, sample_variance = calculate_sample_statistics(sample)
        
        axes[i, j].hist(sample, bins=15, color='green', alpha=0.7, edgecolor='black')
        axes[i, j].set_title(f'Sample {j+1}')
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')
        axes[i, j].text(0.05, 0.95, f'Mean: {sample_mean:.2f}\nVariance: {sample_variance:.2f}',
                        transform=axes[i, j].transAxes, fontsize=10, verticalalignment='top')
        
    axes[i, 0].set_ylabel(f'Sample Size: {sample_size}', fontsize=12)

plt.tight_layout()
plt.show()

# 샘플 평균과 분산의 평균 계산 및 출력
for sample_size in sample_sizes:
    sample_means = results[sample_size]['Sample Means']
    sample_variances = results[sample_size]['Sample Variances']
    mean_of_sample_means = np.mean(sample_means)
    mean_of_sample_variances = np.mean(sample_variances)
    
    print(f'Sample Size: {sample_size}')
    print(f'Mean of Sample Means: {mean_of_sample_means:.2f}')
    print(f'Mean of Sample Variances: {mean_of_sample_variances:.2f}')
    print()
