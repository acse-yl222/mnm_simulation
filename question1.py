from matplotlib import cm
from base_fuctions import *
from parameters1 import *
import matplotlib.animation as anime
import matplotlib.pyplot as plt
import numpy as np
import time

metadata = dict(title="flow", artist="yueyan")
writer = anime.PillowWriter(fps=1, metadata=metadata)


def solve_NS(u, v, p, rho, nu, courant, dt_min, t_end, dx, dy, rtol=1.e-5, logs=False, outint=100):
    """ Solve the incompressible Navier-Stokes equations
    using a lot of the numerical choices and approaches we've seen
    earlier in this lecture.
    """
    t = 0
    u_old = u.copy()
    v_old = v.copy()
    p_old = p.copy()

    velocity_reminder = []
    pressure_reminder = []
    p_RHS = np.zeros_like(X)

    time_it = 0
    while t < t_end:
        '''
        time and step setting
        '''
        time_it += 1
        # set dt based on courant number
        vel_max = np.max(np.sqrt(u ** 2 + v ** 2))
        if vel_max > 0.0:
            dt = min(courant * min(dx, dy) / vel_max, dt_min)
        else:
            dt = dt_min
        t += dt

        '''
        caculate process
        '''
        # calculate intermediate velocities
        u, v = calculate_intermediate_velocity(nu, u, v, u_old, v_old, dt, dx, dy)
        # PPM
        # calculate RHS for the pressure Poisson problem
        p_RHS = calculate_ppm_RHS_central(rho, u, v, p_RHS, dt, dx, dy)
        # compute pressure - note that we use the previous p as an initial guess to the solution
        p = pressure_poisson_jacobi(p_old, dx, dy, p_RHS, 1.e-5, logs=(logs and time_it % outint == 0))
        # project velocity
        u, v = project_velocity(rho, u, v, dt, dx, dy, p)

        velocity_old = np.sqrt(u_old ** 2 + v_old ** 2)
        velocity = np.sqrt(u ** 2 + v ** 2)

        velocity_reminder.append(sl.norm(velocity - velocity_old) / np.maximum(1.0e-10, sl.norm(velocity)))
        pressure_reminder.append(sl.norm(p - p_old) / np.maximum(1.0e-10, sl.norm(p)))

        '''
        print log and plot the graph
        '''
        if logs and time_it % outint == 0:
            print('\nTime = {:.8f}'.format(t))
            print('norm(u) = {0:.8f}, norm(v) = {1:.8f}'.format(sl.norm(u), sl.norm(v)))
            print('Courant number: {0:.8f}'.format(np.max(np.sqrt(u ** 2 + v ** 2)) * dt / min(dx, dy)))
            # set up figure
            fig = plt.figure(figsize=(21, 7))
            ax1 = fig.add_subplot(121)
            cont = ax1.contourf(X, Y, p, cmap=cm.coolwarm)
            fig.colorbar(cont)
            # don't plot at every gird point - every 5th
            ax1.quiver(X[::20, ::5], Y[::20, ::5], u[::20, ::5], v[::20, ::5])
            ax1.set_xlim(-0.001, 0.101)
            ax1.set_ylim(-0.001, 0.051)
            ax1.set_xlabel('$x$', fontsize=16)
            ax1.set_ylabel('$y$', fontsize=16)
            ax1.set_title('Pressure driven problem - pressure and velocity vectors', fontsize=16)

            ax1 = fig.add_subplot(122)
            cont = ax1.contourf(X, Y, np.sqrt(u * u + v * v), cmap=cm.coolwarm)
            fig.colorbar(cont)
            ax1.set_xlim(-0.001, 0.101)
            ax1.set_ylim(-0.001, 0.051)
            ax1.set_xlabel('$x$', fontsize=16)
            ax1.set_ylabel('$y$', fontsize=16)
            ax1.set_title('Pressure driven problem - speed', fontsize=16)
            plt.savefig('question1_plot/pic-{}.png'.format(time_it // outint))
            plt.show()
        # swap pointers without copying data
        temp = u_old
        u_old = u
        u = temp

        temp = v_old
        v_old = v
        v = temp

        temp = p_old
        p_old = p
        p = temp

    return u, v, p, velocity_reminder, pressure_reminder


start = time.time()
u, v, p, velocity_reminder, pressure_reminder = solve_NS(u, v, p, rho, nu, courant, dt_min, t_end, dx, dy, rtol=1.e-5,
                                                         logs=True, outint=1000)
end = time.time()
print('Time taken by calculation = ', end - start)

flow_in = np.sum(u[0, :]) * Ly
flow_out = np.sum(u[-1, :]) * Ly

print('flow in ', flow_in)
print('flow out ', flow_out)

# print the final plot
# set up figure
fig = plt.figure(figsize=(28, 7))
ax1 = fig.add_subplot(131)
cont = ax1.contourf(X, Y, p, cmap=cm.coolwarm)
fig.colorbar(cont)
# don't plot at every gird point - every 5th
ax1.quiver(X[::20, ::5], Y[::20, ::5], u[::20, ::5], v[::20, ::5])
ax1.set_xlim(-0.001, 0.101)
ax1.set_ylim(-0.001, 0.051)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$y$', fontsize=16)
ax1.set_title('Pressure driven problem - pressure and velocity vectors', fontsize=16)

ax1 = fig.add_subplot(132)
cont = ax1.contourf(X, Y, np.sqrt(u * u + v * v), cmap=cm.coolwarm)
fig.colorbar(cont)
ax1.set_xlim(-0.001, 0.101)
ax1.set_ylim(-0.001, 0.051)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$y$', fontsize=16)
ax1.set_title('Pressure driven problem - speed', fontsize=16)

ax1 = fig.add_subplot(133)
ax1.plot(velocity_reminder)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$y$', fontsize=16)
ax1.set_title('velocity reminder', fontsize=16)
plt.show()
