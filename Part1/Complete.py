#Figure 2.1

import numpy as np
import matplotlib.pyplot as plt

def generate_julia_set(c, maxiter=300, width=800, height=800, xmin=-2, xmax=2, ymin=-2, ymax=2):
    """
    Generates and displays a Julia set given a complex constant c.
    
    Parameters:
    - c: Complex number defining the Julia set.
    - maxiter: Maximum number of iterations per point.
    - width, height: Dimensions of the output image.
    - xmin, xmax, ymin, ymax: Limits of the plot in the complex plane.
    """
    iteration_counts = np.zeros((height, width))
    
    for ix in range(width):
        for iy in range(height):
            x = xmin + (xmax - xmin) * ix / (width - 1)
            y = ymin + (ymax - ymin) * iy / (height - 1)
            z = complex(x, y)
            
            for iteration in range(maxiter):
                if abs(z) >= 2.0:
                    break
                z = z*z + c
                
            iteration_counts[iy, ix] = iteration

    plt.figure(figsize=(10, 10))
    plt.imshow(iteration_counts, cmap='gray', extent=(xmin, xmax, ymin, ymax), interpolation='nearest')
    plt.colorbar()
    plt.title(f"Julia Set for c = {c}")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()

# Example usage:
c = -0.62772 - 0.42193j
generate_julia_set(c)

#FIGURE 2.2

import matplotlib.pyplot as plt
import numpy as np

def plot_julia_iterations(c, z_values, maxiter):
    plt.figure(figsize=(10, 6))

    # Iterate over each initial z value
    for z_init in z_values:
        z = z_init
        abs_values = []
        for i in range(maxiter):
            z = z**2 + c
            abs_values.append(abs(z))
            if abs(z) > 2:
                break
        
        # Plot the absolute values
        plt.plot(abs_values, label=f'z={z_init}', marker='o' if z_init == 0 else 'D')
    
    plt.axhline(y=2, color='k', linestyle='--', label='cutoff')
    plt.title('Evolution of abs(z) with c={}'.format(c))
    plt.xlabel('Iteration')
    plt.ylabel('abs(z)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(abs_values) + 0.5)  # adjust the y-axis limit to show all points clearly
    plt.show()

# Constants
c = complex(-0.62772, -0.42193)
maxiter = 50
z_values = [0, -0.82]

# Generate the plot
plot_julia_iterations(c, z_values, maxiter)

#Figure 2.3

import numpy as np
import matplotlib.pyplot as plt
import time

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193

def calculate_z_serial_purepython(maxiter, zs, cs):
    """Calculate output list using Julia update rule"""
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

def calc_pure_python(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex parameters (cs),
    build Julia set"""
    x_step = (x2 - x1) / desired_width
    y_step = (y2 - y1) / desired_width
    x = np.arange(x1, x2, x_step).tolist()
    y = np.arange(y1, y2, y_step).tolist()
    
    # Ensure that zs is defined using the correct x and y variables from the lists
    zs = [complex(xcoord, ycoord) for ycoord in y for xcoord in x]
    cs = [complex(c_real, c_imag) for _ in zs]
    
    print("Length of x:", len(x))
    print("Total elements:", len(zs))
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")
    
    # This sum is expected for a 1000^2 grid with 300 iterations
    assert sum(output) == 33219980, "Calculation did not perform as expected"
    
    # Reshape the output into a 2D array for plotting
    output_2d = np.reshape(output, (len(y), len(x)))
    
    # Plot the Julia set using a grayscale color map
    plt.imshow(output_2d, cmap="gray", extent=(x1, x2, y1, y2))
    plt.colorbar()
    plt.title("Julia set with c = {} + {}j".format(c_real, c_imag))
    plt.show()

# Calculate the Julia set using a pure Python solution with
# reasonable defaults for a laptop
calc_pure_python(desired_width=1000, max_iterations=300)