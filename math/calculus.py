def derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Esempi di funzioni
def quadratic(x):
    return x**2

def exponential(x):
    return 2.71828**x
