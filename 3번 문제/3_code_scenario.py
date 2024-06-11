import numpy as np
import matplotlib.pyplot as plt

# 공통 설정
mu1_base = 50  # 첫 번째 모집단의 평균
mu2_base = 60  # 두 번째 모집단의 평균
sigma_base = 10  # 기준 표준 편차

# 시나리오 1: N 크기 증가
population_sizes = [1500, 15000, 150000]

plt.figure(figsize=(18, 14))
for i, pop_size in enumerate(population_sizes, 1):
    population1 = np.random.normal(mu1_base, sigma_base, pop_size)
    population2 = np.random.normal(mu2_base, sigma_base, pop_size)

    plt.subplot(3, 3, i)
    plt.hist(population1, bins=60, density=True, alpha=0.6, color='b', label='Population 1 Real Data', edgecolor='black')
    plt.hist(population2, bins=60, density=True, alpha=0.6, color='r', label='Population 2 Real Data', edgecolor='black')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p1 = (1 / (sigma_base * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1_base) / sigma_base) ** 2)
    p2 = (1 / (sigma_base * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2_base) / sigma_base) ** 2)
    plt.plot(x, p1, 'b', linewidth=2, label='Population 1 PDF')
    plt.plot(x, p2, 'r', linewidth=2, label='Population 2 PDF')

    plt.title(f'Population Size: {pop_size}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    if i == 1:
        plt.legend()

# 시나리오 2: 평균의 차이 증가
mean_differences = [10, 20, 30]

for i, mean_diff in enumerate(mean_differences, 4):
    mu2 = mu1_base + mean_diff
    population1 = np.random.normal(mu1_base, sigma_base, population_sizes[0])
    population2 = np.random.normal(mu2, sigma_base, population_sizes[0])

    plt.subplot(3, 3, i)
    plt.hist(population1, bins=60, density=True, alpha=0.6, color='b', label='Population 1 Real Data', edgecolor='black')
    plt.hist(population2, bins=60, density=True, alpha=0.6, color='r', label='Population 2 Real Data', edgecolor='black')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p1 = (1 / (sigma_base * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1_base) / sigma_base) ** 2)
    p2 = (1 / (sigma_base * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2) / sigma_base) ** 2)
    plt.plot(x, p1, 'b', linewidth=2, label='Population 1 PDF')
    plt.plot(x, p2, 'r', linewidth=2, label='Population 2 PDF')

    plt.title(f'Mean Difference: {mean_diff}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    if i == 4:
        plt.legend()

# 시나리오 3: 표준 편차 증가
sigmas = [10, 20, 30]

for i, sigma in enumerate(sigmas, 7):
    population1 = np.random.normal(mu1_base, sigma, population_sizes[0])
    population2 = np.random.normal(mu2_base, sigma, population_sizes[0])

    plt.subplot(3, 3, i)
    plt.hist(population1, bins=60, density=True, alpha=0.6, color='b', label='Population 1 Real Data', edgecolor='black')
    plt.hist(population2, bins=60, density=True, alpha=0.6, color='r', label='Population 2 Real Data', edgecolor='black')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p1 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu1_base) / sigma) ** 2)
    p2 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu2_base) / sigma) ** 2)
    plt.plot(x, p1, 'b', linewidth=2, label='Population 1 PDF')
    plt.plot(x, p2, 'r', linewidth=2, label='Population 2 PDF')

    plt.title(f'Standard Deviation: {sigma}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    if i == 7:
        plt.legend()

plt.tight_layout()
plt.show()
