import time
import numpy as np
import scipy.linalg as sl
from matplotlib import cm
import matplotlib.pyplot as plt


def pressure_poisson_jacobi(p, dx, dy, RHS, rtol=1.e-5, logs=False):
    """ Solve the pressure Poisson equation (PPE)
    using Jacobi iteration assuming mesh spacing of
    dx and dy (we assume at the moment that dx=dy)
    and the RHS function given by RHS.

    Assumes imposition of a Neumann BC on all boundaries.

    Return the pressure field.
    """
    # iterate
    tol = 10. * rtol
    it = 0
    p_old = np.copy(p)

    imax = len(p)
    jmax = np.size(p) // imax

    while tol > rtol:
        it += 1
        # this version is valid for dx!=dy
        p[1:-1, 1:-1] = 1.0 / (2.0 + 2.0 * (dx ** 2) / (dy ** 2)) * (p_old[2:, 1:-1] + p_old[:-2, 1:-1] +
                                                                     (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * (
                                                                             dx ** 2) / (dy ** 2)
                                                                     - (dx ** 2) * RHS[1:-1, 1:-1])

        # apply zero gradient Neumann boundary conditions at the no slip walls
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        # p[-1, -(jmax-jmax//2):] = p[-2, -(jmax-jmax//2):]  #不同
        p[-1, jmax // 2:] = p[-2, jmax // 2:]
        # print("My debug: index", p[-1, -(jmax-jmax//2):].shape, p[-1, jmax//2:].shape)
        # relative change in pressure
        tol = sl.norm(p - p_old) / np.maximum(1.0e-10, sl.norm(p))

        # swap arrays without copying the data
        temp = p_old
        p_old = p
        p = temp

    if logs: print('pressure solve iterations = {:4d}'.format(it))
    return p


def calculate_ppm_RHS_upwind(rho, u, v, RHS, dt, dx, dy):
    """ Calculate the RHS of the
    Poisson equation resulting from the projection method.
    Use upwind differences for the first derivatives of u and v.
    """
    RHS[1:-1, 1:-1] = rho * (np.select([u[1:-1, 1:-1] > 0, u[1:-1, 1:-1] <= 0],
                                       [np.diff(u[:-1, 1:-1], n=1, axis=0) / dx,
                                        np.diff(u[1:, 1:-1], n=1, axis=0) / dx]) +
                             np.select([v[1:-1, 1:-1] > 0, v[1:-1, 1:-1] <= 0],
                                       [np.diff(v[1:-1, :-1], n=1, axis=1) / dy,
                                        np.diff(v[1:-1, 1:], n=1, axis=1) / dy]))
    return RHS


def calculate_ppm_RHS_central(rho, u, v, RHS, dt, dx, dy):
    """ Calculate the RHS of the
    Poisson equation resulting from the projection method.
    Use central differences for the first derivatives of u and v.
    """
    RHS[1:-1, 1:-1] = rho * (
            (1.0 / dt) * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
                          + (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)))
    return RHS


def project_velocity(rho, u, v, dt, dx, dy, p):
    """ Update the velocity to be divergence free using the pressure.
    """
    u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt * (1. / rho) * (
            (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx))
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt * (1. / rho) * (
            (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dy))

    return u, v


def calculate_intermediate_velocity(nu, u, v, u_old, v_old, dt, dx, dy):
    """ Calculate the intermediate velocities.
    """
    # intermediate u
    u[1:-1, 1:-1] = u_old[1:-1, 1:-1] - dt * (
        # ADVECTION:  uu_x + vu_x
            u_old[1:-1, 1:-1] *
            # see comments in the upwind based solver for advection above
            np.select([u_old[1:-1, 1:-1] > 0, u_old[1:-1, 1:-1] < 0],
                      [np.diff(u_old[:-1, 1:-1], n=1, axis=0) / dx,
                       np.diff(u_old[1:, 1:-1], n=1, axis=0) / dx]) +
            v_old[1:-1, 1:-1] *
            np.select([v_old[1:-1, 1:-1] > 0, v_old[1:-1, 1:-1] < 0],
                      [np.diff(u_old[1:-1, :-1], n=1, axis=1) / dy,
                       np.diff(u_old[1:-1, 1:], n=1, axis=1) / dy])) + (
                        # DIFFUSION
                            dt * nu * (np.diff(u_old[:, 1:-1], n=2, axis=0) / (dx ** 2)
                                       + np.diff(u_old[1:-1, :], n=2, axis=1) / (dy ** 2)))
    # intermediate v
    v[1:-1, 1:-1] = v_old[1:-1, 1:-1] - dt * (
        # ADVECTION:  uv_x + vv_x
            u_old[1:-1, 1:-1] *
            np.select([u_old[1:-1, 1:-1] > 0, u_old[1:-1, 1:-1] < 0],
                      [np.diff(v_old[:-1, 1:-1], n=1, axis=0) / dx,
                       np.diff(v_old[1:, 1:-1], n=1, axis=0) / dx]) +
            v_old[1:-1, 1:-1] *
            np.select([v_old[1:-1, 1:-1] > 0, v_old[1:-1, 1:-1] < 0],
                      [np.diff(v_old[1:-1, :-1], n=1, axis=1) / dy,
                       np.diff(v_old[1:-1, 1:], n=1, axis=1) / dy])) + (
                        # DIFFUSION
                            dt * nu * (np.diff(v_old[:, 1:-1], n=2, axis=0) / (dx ** 2)
                                       + np.diff(v_old[1:-1, :], n=2, axis=1) / (dy ** 2)))

    # apply the velocity boundary condition to the intermediate velocity data
    imax = len(u)
    jmax = np.size(u) // imax
    u[0, :] = u[1, :]
    u[-1, :jmax // 2] = u[-2, :jmax // 2]

    return u, v


def calculate_advection(c, u, v, dx, dy):
    A = np.zeros_like(c)
    A[1:-1, 1:-1] = u[1:-1, 1:-1] * np.select([u[1:-1, 1:-1] >= 0, u[1:-1, 1:-1] < 0],
                                              [np.diff(c[:-1, 1:-1], n=1, axis=0) / dx,
                                               np.diff(c[1:, 1:-1], n=1, axis=0) / dx])
    + v[1:-1, 1:-1] * np.select([c[1:-1, 1:-1] >= 0, c[1:-1, 1:-1] < 0],
                                [np.diff(c[1:-1, :-1], n=1, axis=1) / dy,
                                 np.diff(c[1:-1, 1:], n=1, axis=1) / dy])
    # 应该不用加BC
    return A


def calculate_diffusion(c, dx, dy):
    diffusion = np.zeros_like(c)
    diffusion[1:-1, 1:-1] = (c[2:, 1:-1] - 2 * c[1:-1, 1:-1] + c[:-2, 1:-1]) / (dx ** 2) + (
                c[1:-1, 2:] - 2 * c[1:-1, 1:-1] + c[1:-1, :-2]) / (dy ** 2)
    # 应该不用加BC
    return diffusion


def calculate_c(c, u, v, D, k, dt, dx, dy):
    c_old = c.copy()
    c[1:-1, 1:-1] = - calculate_advection(c_old, u, v, dx, dy)[1:-1, 1:-1] * dt + D * calculate_diffusion(c_old, dx,
                                                                                                          dy)[1:-1,
                                                                                      1:-1] * dt - k * c_old[1:-1,
                                                                                                       1:-1] * dt + c_old[
                                                                                                                    1:-1,
                                                                                                                    1:-1]
    # 加BC
    c[0, :] = 1
    c[:, 0] = p[:, 1]
    c[:, -1] = c[:, -2]
    c[-1, :] = c[-2, :]
    return c


def solve_NS_with_concentration_PDE(u, v, X, c, p, rho, nu, D, k, courant, dt_min, t_end, dx, dy, rtol=1.e-5,
                                    logs=False, outint=100):  # 无改动
    """ Solve the incompressible Navier-Stokes equations
    using a lot of the numerical choices and approaches we've seen
    earlier in this lecture.
    """
    t = 0

    u_old = u.copy()
    v_old = v.copy()
    p_RHS = np.zeros_like(X)

    time_it = 0
    # print("Here2")
    while t < t_end:
        # print("Here3", time_it, outint)
        time_it += 1
        # set dt based on courant number
        vel_max = np.max(np.sqrt(u ** 2 + v ** 2))
        if vel_max > 0.0:
            dt = min(courant * min(dx, dy) / vel_max, dt_min)
        else:
            dt = dt_min

        t += dt
        if logs and time_it % outint == 0:
            print('\nTime = {:.8f}'.format(t))

        # print("Here4")
        # calculate intermediate velocities
        u, v = calculate_intermediate_velocity(nu, u, v, u_old, v_old, dt, dx, dy)

        # print("Here5")
        # PPM
        # calculate RHS for the pressure Poisson problem
        p_RHS = calculate_ppm_RHS_central(rho, u, v, p_RHS, dt, dx, dy)

        # print("Here6")
        # compute pressure - note that we use the previous p as an initial guess to the solution
        p = pressure_poisson_jacobi(p, dx, dy, p_RHS, 1.e-5, logs=(logs and time_it % outint == 0))

        c = calculate_c(c, u, v, D, k, dt, dx, dy)
        # print("Here7")
        # project velocity
        u, v = project_velocity(rho, u, v, dt, dx, dy, p)  # 下一轮新速度

        if logs and time_it % outint == 0:
            print('norm(u) = {0:.8f}, norm(v) = {1:.8f}'.format(sl.norm(u), sl.norm(v)))
            print('Courant number: {0:.8f}'.format(np.max(np.sqrt(u ** 2 + v ** 2)) * dt / min(dx, dy)))
            fig = plt.figure(figsize=(21, 7))
            ax1 = fig.add_subplot(131)
            cont = ax1.contourf(X, Y, p, cmap=cm.coolwarm)
            fig.colorbar(cont)
            # don't plot at every gird point - every 5th
            # ax1.quiver(X[::10,::5],Y[::10,::5],u[::10,::5],v[::10,::5], angles='xy', scale_units='xy', scale=20)
            ax1.quiver(X[::10, ::5], Y[::10, ::5], u[::10, ::5], v[::10, ::5])
            ax1.set_xlim(0, 0.1)
            ax1.set_ylim(0, 0.05)
            ax1.set_xlabel('$x$', fontsize=16)
            ax1.set_ylabel('$y$', fontsize=16)
            ax1.set_title('Pressure driven problem - pressure and velocity vectors', fontsize=16)

            ax1 = fig.add_subplot(132)
            cont = ax1.contourf(X, Y, np.sqrt(u * u + v * v), cmap=cm.coolwarm)
            fig.colorbar(cont)
            ax1.set_xlim(0, 0.1)
            ax1.set_ylim(0, 0.05)
            ax1.set_xlabel('$x$', fontsize=16)
            ax1.set_ylabel('$y$', fontsize=16)
            ax1.set_title('Pressure driven problem - speed', fontsize=16)

            ax1 = fig.add_subplot(133)
            cont = ax1.contourf(X, Y, c, cmap=cm.coolwarm)
            fig.colorbar(cont)
            ax1.set_xlim(0, 0.1)
            ax1.set_ylim(0, 0.05)
            ax1.set_xlabel('$x$', fontsize=16)
            ax1.set_ylabel('$y$', fontsize=16)
            ax1.set_title('Pressure driven problem - concentration', fontsize=16)
            plt.show()

        # swap pointers without copying data
        temp = u_old
        u_old = u
        u = temp
        temp = v_old
        v_old = v
        v = temp
    return u, v, p, c


# physical parameters
rho = 1e3
mu = 1e-3
nu = mu / rho

# define spatial mesh
# Size of rectangular domain
Lx = 0.1
Ly = 0.05

P_max = 0.5
c_init = 1
D = 1e-6
k = 1.

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
c = np.zeros_like(X)

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
p[-1, :jmax // 2] = 0.0
c[0, :] = c_init

# set a Courant number and use dynamic time step
courant = 0.005
dt_min = 1.e-3

t_end = 10.0

start = time.time()
# print("Here1")
u, v, p, c = solve_NS_with_concentration_PDE(u, v, X, c, p, rho, nu, D, k, courant, dt_min, t_end, dx, dy, rtol=1.e-6,
                                             logs=True, outint=1000)
end = time.time()
print('Time taken by calculation = ', end - start)

# set up figure
