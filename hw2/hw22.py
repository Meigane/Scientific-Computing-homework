# hw22.py
# 学号-姓名

import numpy as np

def PLU_JKI(A):
    """列主元 JKI 型 LU 分解"""
    n = A.shape[0]
    A = A.copy()
    p = list(range(n))
    
    for j in range(n-1):
        # 找主元
        pivot = j + np.argmax(np.abs(A[j:n, j]))
        
        # 交换行
        if pivot != j:
            A[[j, pivot], :] = A[[pivot, j], :]
            p[j], p[pivot] = p[pivot], p[j]
        
        # 消元
        A[j+1:n, j] = A[j+1:n, j] / A[j, j]
        A[j+1:n, j+1:n] = A[j+1:n, j+1:n] - np.outer(A[j+1:n, j], A[j, j+1:n])
    
    # 提取 L 和 U
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    
    return L, U, np.array(p)


# ==================== 主程序 ====================
print("读取数据...")
data = np.loadtxt('hw2\hw22_data.txt', delimiter=',')
print(f"矩阵大小: {data.shape}")

print("进行LU分解...")
L, U, p = PLU_JKI(data)
print("LU分解完成")

print("计算误差（逐行计算）...")
n = data.shape[0]
A_permuted = data[p, :]

# 而是逐行计算L@U
diff_norm_sq = 0.0
data_norm_sq = 0.0

for i in range(n):
    if i % 10 == 0:
        print(f"  进度: {i}/{n}")
    
    # 计算 (L@U) 的第 i 行：用向量内积
    LU_row_i = np.zeros(n)
    for k in range(n):
        LU_row_i += L[i, k] * U[k, :]
    
    # 计算该行的误差
    diff_row = A_permuted[i, :] - LU_row_i
    diff_norm_sq += np.sum(diff_row ** 2)
    
    # 同时计算原矩阵的范数
    data_norm_sq += np.sum(data[i, :] ** 2)

relerr = np.sqrt(diff_norm_sq) / np.sqrt(data_norm_sq)

print(f'\nrelerr = {relerr:.4e}')