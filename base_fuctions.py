import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl


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
        p[1:-1, 1:-1] = 1.0 / (2.0 + 2.0 * (dx ** 2) / (dy ** 2)) * (p_old[2:, 1:-1] +
                                                                     p_old[:-2, 1:-1] +
                                                                     (p_old[1:-1, 2:] + p_old[1:-1, :-2]) *
                                                                     (dx ** 2) / (dy ** 2) - (dx ** 2) * RHS[1:-1,
                                                                                                         1:-1])
        # todo: change the boundary condition
        # apply zero gradient Neumann boundary conditions at the no slip walls
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[-1, jmax // 2:] = p[-2, jmax // 2:]

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
    RHS[1:-1, 1:-1] = rho * ((1.0 / dt) * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
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

    # todo: change the boundary condition
    # apply the velocity boundary condition to the intermediate velocity data
    # apply the velocity boundary condition
    imax = len(u)
    jmax = np.size(u) // imax
    u[0, :] = u[1, :]
    u[-1, :jmax // 2] = u[-2, :jmax // 2]
    return u, v


def calculate_C_diffusion_term_RHS(C, dx, dy, D, dt):
    RHS = dt * D * (
            (C[0:-2, 1:-1] + C[2:, 1:-1] - 2 * C[1:-1, 1:-1]) / dx ** 2 +
            (C[1:-1, 0:-2] + C[1:-1, 2:] - 2 * C[1:-1, 1:-1]) / dy ** 2)
    return RHS


def caculate_C_advection_term_RHS(C, dx, dy, u, v, dt):
    RHS = dt * (u[1:-1, 1:-1]*np.select([u[1:-1, 1:-1] > 0, u[1:-1, 1:-1] <= 0],
                          [np.diff(C[:-1, 1:-1], n=1, axis=0) / dx,
                           np.diff(C[1:, 1:-1], n=1, axis=0) / dx]) +
                v[1:-1, 1:-1]*np.select([v[1:-1, 1:-1] > 0, v[1:-1, 1:-1] <= 0],
                          [np.diff(C[1:-1, :-1], n=1, axis=1) / dy,
                           np.diff(C[1:-1, 1:], n=1, axis=1) / dy]))
    return RHS


def caculate_C_last_term_RHS(C, dt, k):
    RHS = C[1:-1, 1:-1] * dt * k
    return RHS


def caculate_C(C, dx, dy, u, v, dt, k, D):
    C_new = copy.copy(C)
    advection_term = caculate_C_advection_term_RHS(C, dx, dy, u, v,dt)
    diffusion_term = calculate_C_diffusion_term_RHS(C, dx, dy, D,dt)
    last_term = caculate_C_last_term_RHS(C, dt, k)
    C_new[1:-1, 1:-1] = C[1:-1, 1:-1] - advection_term + diffusion_term - last_term
    C_new[0, :] = 1
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]
    return C_new
