"""Julia set generator without optional PIL-based image drawing"""
import time
import numpy as np
import julia3

# area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -.42193



def calc_julia(desired_width, max_iterations):
    """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""
    x_step = (float(x2 - x1) / float(desired_width))
    y_step = (float(y1 - y2) / float(desired_width))
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our
    # function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))
  
    zs_np = np.array(zs, np.complex128)
    cs_np = np.array(cs, np.complex128)

    print ("Length of x:", len(x))
    print ("Total elements:", len(zs))
    start_time = time.time()
    output = julia3.calculate_z(max_iterations, zs_np, cs_np)
    end_time = time.time()
    secs = end_time - start_time
    print ("calculate_z_serial_purepython took", secs, "seconds")

    # this sum is expected for 1000^2 grid with 300 iterations
    #assert sum(output) == 33219980


if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop
    # set draw_output to True to use PIL to draw an image
    calc_julia(desired_width=1000, max_iterations=300)
