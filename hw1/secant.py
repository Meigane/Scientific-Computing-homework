def f(x):
    """define the funtion"""
    return 1 / (x - 1) + 1 / (x - 5) - 1 / 3 

def secant_iter(x0, x1):
    return x1 - (x1 - x0) * f(x1) / (f(x1) - f(x0))



def secant(x0, x1, tol, itermax):
    if abs(f(x0)) < tol:
        return x0, 0
    if abs(f(x1)) < tol:
        return x1, 0
    
    for k in range(1, itermax + 1):
        x2 = secant_iter(x0, x1)
        if abs(x2 - x1) < tol or abs(f(x2)) < tol:
            return x2 , k
        x0 = x1
        x1 = x2
        
    return None, None

xt  = 2.394448724536011  # precise reference value
tol = 1e-6   # precision requirment
itermax = 30 # maxium iterition
# solving range
a = 1.1
b = 4.9 
x0 = 1.1 
x1 = 3.0 # 迭代初值

x, iter = secant(x0, x1, tol, itermax)
print(f'Secant 迭代：近似解 x={x}, 迭代步数{iter}')