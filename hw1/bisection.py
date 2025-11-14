def f(x):
    """define the funtion"""
    return x*x + x - 1

def bisection(a, b, tol, itermax):
    fa = f(a)
    fb = f(b)
    
    if abs(fa) < tol:
        return a, 0
    if abs(fb) < tol:
        return b, 0
    if fa * fb > 0:
        return None, None
    
    for i in range(1, itermax + 1):
        x = 0.5 * (a + b)
        fx = f(x)
        
        if abs(fx) < tol or abs(b - a) < tol:
            return x, i
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
    return None, None


tol = 0.005   # precision requirment
itermax = 30 # maxium iterition
# solving range
a = 0
b = 1

# 调用二分法 
x, iter = bisection(a,b,tol,itermax)
print(f'二分法：近似解 {x}, 迭代步数{iter}')

print(f(x))