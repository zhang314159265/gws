def cdiv(a, b):
    return (a + b - 1) // b

def next_power_of_two(x):
    if x & -x == x:
        return x
    while x != (x & -x):
        x = x & (x - 1)
    return x * 2


