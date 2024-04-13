import time
from functools import wraps

# Function without decorator for direct timing
def calculate_z_serial_purepython_direct(maxiter, zs, cs):
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output

# Decorated function
def calculate_z_serial_purepython(maxiter, zs, cs):
    return calculate_z_serial_purepython_direct(maxiter, zs, cs)

if __name__ == "__main__":
    # Global constants for the Julia set calculation
    x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
    c_real, c_imag = -0.62772, -0.42193

    desired_width = 1000
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    zs = [complex(x1 + i*x_step, y1 + j*y_step) for j in range(desired_width) for i in range(desired_width)]
    cs = [complex(c_real, c_imag) for _ in range(desired_width**2)]

    print(f"Length of x: {desired_width}")
    print(f"Total elements: {len(zs)}")

    # Direct timing
    start_time = time.time()
    calculate_z_serial_purepython_direct(300, zs, cs)
    end_time = time.time()
    print(f"calculate_z_serial_purepython took {end_time - start_time} seconds")

    # Decorated timing
    calculate_z_serial_purepython(300, zs, cs)