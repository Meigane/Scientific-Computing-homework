# 10210720404-张凌菱(python)
# Numerical Methods for Finding Roots of f(x) = 1/(x-1) + 1/(x-5) - 1/3

# ============================================================================
# Function Definitions
# ============================================================================

def f(x):
    """Target function: f(x) = 1/(x-1) + 1/(x-5) - 1/3"""
    return 1 / (x - 1) + 1 / (x - 5) - 1 / 3


def df(x):
    """Derivative: f'(x) = -1/(x-1)^2 - 1/(x-5)^2"""
    return -(1 / (x - 1) ** 2 + 1 / (x - 5) ** 2)


# ============================================================================
# Bisection Method
# ============================================================================

def bisection(a, b, tol, max_iter):
    """
    Bisection method for finding roots in interval [a, b].
    
    Returns: (root, iterations) or (None, None) if fails
    """
    fa, fb = f(a), f(b)
    
    # Check if endpoints are already roots
    if abs(fa) < tol:
        return a, 0
    if abs(fb) < tol:
        return b, 0
    
    # Check if root exists in interval
    if fa * fb > 0:
        return None, None
    
    # Main bisection loop
    for i in range(1, max_iter + 1):
        x = 0.5 * (a + b)
        fx = f(x)
        
        # Check convergence
        if abs(fx) < tol or abs(b - a) < tol:
            return x, i
        
        # Update interval
        if f(a) * fx < 0:
            b = x
        else:
            a = x
    
    return None, None


# ============================================================================
# Newton's Method
# ============================================================================

def newton_iter(x):
    """Single Newton iteration: x - f(x)/f'(x)"""
    return x - f(x) / df(x)


def newton(x0, tol, max_iter):
    """
    Newton's method with initial guess x0.
    
    Returns: (root, iterations) or (None, None) if fails
    """
    if abs(f(x0)) < tol:
        return x0, 0
    
    for k in range(1, max_iter + 1):
        x1 = newton_iter(x0)
        
        # Check convergence
        if abs(x1 - x0) < tol or abs(f(x1)) < tol:
            return x1, k
        
        x0 = x1
    
    return None, None


# ============================================================================
# Steffensen's Acceleration Method
# ============================================================================

def steffensen(x, phi):
    """
    Steffensen's acceleration formula:
    ψ(x) = x - (φ(x) - x)² / (φ(φ(x)) - 2φ(x) + x)
    """
    phi_x = phi(x)
    phi_phi_x = phi(phi_x)
    return x - (phi_x - x) ** 2 / (phi_phi_x - 2 * phi_x + x)


def newton_steffensen(x0, tol, max_iter):
    """
    Newton's method accelerated by Steffensen's iteration.
    
    Returns: (root, iterations) or (None, None) if fails
    """
    if abs(f(x0)) < tol:
        return x0, 0
    
    for k in range(1, max_iter + 1):
        x1 = steffensen(x0, newton_iter)
        
        # Check convergence
        if abs(x1 - x0) < tol or abs(f(x1)) < tol:
            return x1, k
        
        x0 = x1
    
    return None, None


# ============================================================================
# Secant Method
# ============================================================================

def secant_iter(x0, x1):
    """Single secant iteration"""
    return x1 - (x1 - x0) * f(x1) / (f(x1) - f(x0))


def secant(x0, x1, tol, max_iter):
    """
    Secant method with two initial guesses x0 and x1.
    
    Returns: (root, iterations) or (None, None) if fails
    """
    if abs(f(x0)) < tol:
        return x0, 0
    if abs(f(x1)) < tol:
        return x1, 0
    
    for k in range(1, max_iter + 1):
        x2 = secant_iter(x0, x1)
        
        # Check convergence
        if abs(x2 - x1) < tol or abs(f(x2)) < tol:
            return x2, k
        
        x0, x1 = x1, x2
    
    return None, None


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Parameters
    xt = 2.394448724536011  # Precise reference value
    tol = 1e-6              # Tolerance
    max_iter = 30           # Maximum iterations
    
    print("=" * 60)
    print("Root Finding for f(x) = 1/(x-1) + 1/(x-5) - 1/3 = 0")
    print("=" * 60)
    print(f"Reference value: {xt}")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {max_iter}\n")
    
    # Method 1: Bisection Method
    a, b = 1.1, 4.9
    x, iterations = bisection(a, b, tol, max_iter)
    print(f"Bisection Method:    x = {x:.12f}, iterations = {iterations}")
    
    # Method 2: Newton's Method
    x0 = 3.0
    x, iterations = newton(x0, tol, max_iter)
    print(f"Newton's Method:     x = {x:.12f}, iterations = {iterations}")
    
    # Method 3: Steffensen's Acceleration
    x0 = 3.0
    x, iterations = newton_steffensen(x0, tol, max_iter)
    print(f"Steffensen's Method: x = {x:.12f}, iterations = {iterations}")
    
    # Method 4: Secant Method
    x0, x1 = 1.1, 3.0
    x, iterations = secant(x0, x1, tol, max_iter)
    print(f"Secant Method:       x = {x:.12f}, iterations = {iterations}")
    
    print("=" * 60)