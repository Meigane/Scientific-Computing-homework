# 10210720404-张凌菱
# Jacobi and CG for Hilbert linear equations

import numpy as np
from scipy.linalg import hilbert

def sor(A, b, x, tol, IterMax, omega):
    """
    SOR (Successive Over-Relaxation) 迭代法求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵 (n×n)
        b: 右端项向量 (n×1)
        x: 初始解向量 (n×1)
        tol: 收敛容差
        IterMax: 最大迭代次数
        omega: 松弛参数
    
    返回:
        x: 迭代解
        iter: 实际迭代次数
        relres: 相对残量
        flag: 收敛标记 (1=收敛, 0=不收敛)
    """
    n = len(b)
    x = x.copy()  # 避免修改输入
    b_norm = np.linalg.norm(b)
    
    for iter in range(1, IterMax + 1):
        x_old = x.copy()
        
        # SOR 迭代
        for i in range(n):
            # 计算 sigma1 = sum(a_ij * x_j^(k+1)) for j < i
            sigma1 = np.dot(A[i, :i], x[:i])
            
            # 计算 sigma2 = sum(a_ij * x_j^(k)) for j > i
            sigma2 = np.dot(A[i, i+1:], x_old[i+1:])
            
            # SOR 更新公式
            x[i] = (1 - omega) * x_old[i] + omega / A[i, i] * (b[i] - sigma1 - sigma2)
        
        # 计算相对残量
        residual = b - A @ x
        relres = np.linalg.norm(residual) / b_norm
        
        # 检查收敛
        if relres < tol:
            flag = 1
            return x, iter, relres, flag
    
    # 未收敛
    flag = 0
    return x, IterMax, relres, flag


def CG(A, b, x, tol, IterMax):
    """
    CG (Conjugate Gradient) 共轭梯度法求解线性方程组 Ax = b
    
    参数:
        A: 系数矩阵 (n×n，必须是对称正定矩阵)
        b: 右端项向量 (n×1)
        x: 初始解向量 (n×1)
        tol: 收敛容差
        IterMax: 最大迭代次数
    
    返回:
        x: 迭代解
        iter: 实际迭代次数
        relres: 相对残量
        flag: 收敛标记 (1=收敛, 0=不收敛)
    """
    x = x.copy()  # 避免修改输入
    b_norm = np.linalg.norm(b)
    
    # 初始化
    r = b - A @ x  # 初始残量
    p = r.copy()   # 初始搜索方向
    
    for iter in range(1, IterMax + 1):
        # 计算 alpha_k
        Ap = A @ p
        r_dot_r = np.dot(r, r)
        alpha = r_dot_r / np.dot(p, Ap)
        
        # 更新 x
        x = x + alpha * p
        
        # 更新残量 r
        r_new = r - alpha * Ap
        
        # 计算相对残量
        relres = np.linalg.norm(r_new) / b_norm
        
        # 检查收敛
        if relres < tol:
            flag = 1
            return x, iter, relres, flag
        
        # 计算 beta_k
        r_new_dot_r_new = np.dot(r_new, r_new)
        beta = r_new_dot_r_new / r_dot_r
        
        # 更新搜索方向 p
        p = r_new + beta * p
        
        # 更新 r
        r = r_new
    
    # 未收敛
    flag = 0
    return x, IterMax, relres, flag


# 主程序
if __name__ == "__main__":
    # 参数设置
    n = 32
    H = hilbert(n)          # Hilbert 矩阵
    xtrue = np.ones(n)      # 真解
    b = H @ xtrue           # 右端项
    tol = 1e-4              # 收敛容差
    IterMax = 200           # 最大迭代次数
    x0 = np.zeros(n)        # 初始解
    omega = 1.5             # SOR 松弛参数
    
    # SOR 迭代法
    x_sor, iter_sor, relres_sor, flag_sor = sor(H, b, x0, tol, IterMax, omega)
    
    if flag_sor == 1:
        print(f'SOR: Iter={iter_sor}, relres={relres_sor:.4e}')
    else:
        print(f'SOR: 不收敛！relres={relres_sor:.4e}')
    
    # CG 迭代法
    x_cg, iter_cg, relres_cg, flag_cg = CG(H, b, x0, tol, IterMax)
    
    if flag_cg == 1:
        print(f'CG : Iter={iter_cg}, relres={relres_cg:.4e}')
    else:
        print(f'CG : 不收敛！relres={relres_cg:.4e}')
