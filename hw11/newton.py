def f(x):
    """define the funtion"""
    return 1 / (x - 1) + 1 / (x - 5) - 1 / 3 

def df(x):
    return - (1 / (x - 1) ** 2 + 1 / (x - 5) ** 2)

def newton_iter(x0):
    return x0 - f(x0) / df(x0)

xt  = 2.394448724536011  # precise reference value
tol = 1e-6   # precision requirment
itermax = 30 # maxium iterition
# solving range
a = 1.1
b = 4.9 

x0 = 3.0    # 迭代初值


def newton(x0, tol, itermax):
    if abs(f(x0)) < tol:
        return x0 , 0
    
    for k in range(1, itermax + 1):
        x1 = newton_iter(x0)
        if abs(x1 - x0) < tol or abs(f(x1)) < tol:
            return x1 , k
        x0 = x1
        
    return None, None

x, iter = newton(x0, tol, itermax)
print(f'Newton 迭代：近似解 x={x}, 迭代步数{iter}')