# hw21.py
# 学号-姓名

import numpy as np

def modified_cholesky(H):
    """
    改进的平方根法: H = L*D*L^T
    L存放在H的下三角, D存放在H的对角
    """
    n = H.shape[0]
    A = H.copy()
    
    # 算法 改进的平方根法 I
    for j in range(n):
        # 计算 a_jj = a_jj - sum(a_jk^2 * a_kk)
        for k in range(j):
            A[j,j] = A[j,j] - A[j,k]**2 * A[k,k]
        
        # 计算第j列
        for i in range(j, n):
            for k in range(j):
                A[i,j] = A[i,j] - A[i,k] * A[k,k] * A[j,k]
            A[i,j] = A[i,j] / A[j,j]
    
    return A

def solve_Ly_b(A, b):
    """
    向前回代求解 Ly = b
    L存放在A的下三角(对角线为1)
    """
    n = len(b)
    y = np.zeros(n)
    
    # 算法 改进的平方根法 II (前向代入部分)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i]
        for j in range(i):
            y[i] = y[i] - A[i,j] * y[j]
    
    return y


def solve_DLTx_y(A, y):
    """
    向后回代求解 D*L^T*x = y
    D存放在A的对角, L存放在A的下三角
    """
    n = len(y)
    x = np.zeros(n)
    
    # 算法 改进的平方根法 II (后向代入部分)
    x[n-1] = y[n-1] / A[n-1,n-1]
    for i in range(n-2, -1, -1):
        x[i] = y[i] / A[i,i]
        for j in range(i+1, n):
            x[i] = x[i] - A[j,i] * x[j]
    
    return x

# 主程序
n = 16
H = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])
xtrue = np.ones(n)
b = H @ xtrue

# 改进的平方根分解
A = modified_cholesky(H)

# 解方程 Ly = b
y = solve_Ly_b(A, b)

# 解方程 DL^T x = y
x = solve_DLTx_y(A, y)

# 输出结果
relres = np.linalg.norm(b - H @ x) / np.linalg.norm(b)
relerr = np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue)

print(f'相对残量为: {relres:.4e}')
print(f'相对误差为: {relerr:.4e}')