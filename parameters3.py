import numpy as np

# physical parameters
rho = 1000
mu = 0.001
nu = mu / rho

# define spatial mesh
# Size of rectangular domain
Lx = 0.1
Ly = 0.05

P_max = 0.5

# Number of grid points in each direction, including boundary nodes
Nx = 101
Ny = 51

# hence the mesh spacing
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# read the docs to see the ordering that mgrid gives us
X, Y = np.mgrid[0:Nx:1, 0:Ny:1]
X = dx * X
Y = dy * Y
# the following is an alternative to the three lines above
# X, Y = np.mgrid[0: Lx + 1e-10: dx, 0: Ly + 1e-10: dy]
# but without the need to add a "small" increement to ensure
# the Lx and Ly end points are included

# initialise independent variables
u = np.zeros_like(X)
v = np.zeros_like(X)
p = np.zeros_like(X)

# Apply Dirichlet BCs to u and v - the code below doesn't touch
# these so we can do this once outside the time loop
u[:, -1] = 0
u[:, 0] = 0
u[0, :] = 0
u[-1, :] = 0
v[:, -1] = 0
v[:, 0] = 0
v[0, :] = 0
v[-1, :] = 0
imax = len(p)
jmax = np.size(p) // imax
p[0, :] = P_max
p[-1, (jmax // 2):] = 0.0

# set a Courant number and use dynamic time step
courant = 0.005
dt_min = 1.e-3

t_end = 10

C = np.zeros_like(X)
C[0, :] = 1

k = 1
D = 1e-7


