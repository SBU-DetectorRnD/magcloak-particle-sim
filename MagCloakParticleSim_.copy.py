#!/usr/bin/env python3

"""Calculates the path of a particle in a magnetic field.

This code was written for Python 3 and tested using Python 3.5.0
(64-bit) with Anaconda 2.4.0 (64-bit) on a computer running Ubuntu 14.04
LTS (64-bit).
"""

__author__ = 'Kyle Capobianco-Hogan'
__copyright = 'Copyright 2016'
__credits__ = ['Kyle Capobianco-Hogan']
#__license__
__version__ = '0.0.0'
__maintainer__ = 'Kyle Capobianco-Hogan'
__email__ = 'kylech.git@gmail.com'
__status__ = 'Development'

# ======================================================================
# Import modules.
# ======================================================================

# Standard library modules.

# Third party modules.
import numpy as np
import scipy.constants as const

from coordinate_converter import *

# ======================================================================
# Constants.
# ======================================================================

c = const.c

m = const.m_p
m_MeV = const.value('proton mass energy equivalent in MeV')

qpm = const.value('proton charge to mass quotient')

q = const.e

eV = const.eV

# ======================================================================
# Main.
# ======================================================================

def main():
    # Print script name, file name, and version.
    print('\n'
        + '='*80 + '\n'
        + '\n'
        + 'Magnetic Cloak Particle Simulation'.center(80, ' ') + '\n'
        #+ ('(' + __file__ + ')').center(80, ' ')
        + __file__.center(80, ' ')
        + ('v ' + __version__).center(80, ' ') + '\n'
        + '\n'
        + '='*80 + '\n')
    
    # Initialize base method dictionary.
    base_methods_dict = {'0' : in_terminal,
                         '1' : from_batch,
                         '2' : GUI}
    
    # Prompt user to specify method of running the program.
    print('Methods of using ' + __file__ + ':\n'
        + '    0 : to enter parameters in the terminal.\n'
        + '    1 : *to use a batch file.*\n'
        + '    2 : *to use the GUI.*\n'
        + '    * Is not currently implemented.')
    base_methods_dict[input(
        'How do you want to run ' + __file__ + '?\n>>> ')]()
    return 0

# ======================================================================
# Terminal interface function.
# ======================================================================

def in_terminal():
    
    # ==================================================================
    # Prompt user to input simulation parameters.
    # ==================================================================
    
    # ------------------------------------------------------------------
    # Differential equation.
    # ------------------------------------------------------------------
    
    print('\n'
        + 'Available differential equations.\n'
        + '    mag : Lorentz force from magnetic field.')
    
    diff_eq_key = input(
        'Which differential equation do you want to use?\n>>> ')
    print()
    f = diff_eq_dict[diff_eq_key]
    
    # ------------------------------------------------------------------
    # Fields required by differential equation.
    # ------------------------------------------------------------------
    
    # Magnetic field.
    if diff_eq_key in ('mag'):
        B = B_maker()
    print()
    
    # ------------------------------------------------------------------
    # Solution approximation method.
    # ------------------------------------------------------------------
    
    print('Available solution approximation methods.'
        + '\n    Eul : Euler method.'
        + '\n    RK2 : Second order Runge_Kutta method (RK2).'
        + '\n    RK4 : Fourth order Runge-Kutta method (RK4).')
    step_key = input(
        'Which solution approximation method do you want to use?\n>>> ')
    print()
    step = sol_approx_method_dict[step_key]
    
    # ------------------------------------------------------------------
    # Tracking settings.
    # ------------------------------------------------------------------
    
    print('Tracking interval options.'
        + '\n    0 : run without tracking.'
        + '\n    n : (integer > 0) record every n-th step.')
    trackint = int(input(
        'What tracking interval would you like to use?\n>>> '))
    print()
    
    # ------------------------------------------------------------------
    # Maximum number of iterations.
    # ------------------------------------------------------------------
    
    nmax = int(input(
        'What is the maximum number of iterations that should be run?\n>>> '))
    if nmax <= 0:
        print('Error: You set the maximum number of iterations to run (nmax) '
            + 'to a value less than or equal to zero. The maximum number of '
            + 'iterations must be a positive integer value (type int); the '
            + 'program will attempt to convert a non-integer entry to an '
            + 'integer using the built-in function int.')
        raise SystemExit
    print()
    
    # ------------------------------------------------------------------
    # Iteration exit conditions.
    # ------------------------------------------------------------------
    
    print('Available exit conditions:'
        + '\n    0 : none, will run nmax iterations.'
        #+ '\n    box : a box of user specified dimensions.'
        )
    exitcond_key = input('Which exit condition would you like to use?\n>>> ')
    # <=========================================================================
    print()
    
    # ------------------------------------------------------------------
    # Initial conditions.
    # ------------------------------------------------------------------
    
    t0, y0 = init_cond_setup_dict[diff_eq_key]()
    print()
    
    # ------------------------------------------------------------------
    # Step size.
    # ------------------------------------------------------------------
    
    #print('Ways of specifying step size available:'
    #    + '\n    dt : a time interval.'
    #   #+ '\n    dl : a distance (converted to time based on initial speed).'
    #    )
    #h_type = input('How would you like to specify step size?\n>>> ')
    #h_type_dict[h_type]
    h = float(input('Select step size [s]:\n>>> '))
    print()
    
    # ==================================================================
    # Run simulation.
    # ==================================================================
    
    iterator(
        h,
        t=t0,
        y=y0,
        nmax=nmax,
        #exitcond=exitcond,
        trackint=trackint
        )

# ======================================================================
# Batch file "interface" function.
# ======================================================================

def from_batch():
    print('I am sorry.  This feature is not currently implemented.')
    pass

# ======================================================================
# GUI function.
# ======================================================================

def GUI():
    print('I am sorry.  This feature is not currently implemented.')
    pass

# ======================================================================
# Differential equation functions and dictionary.
# ======================================================================

# ----------------------------------------------------------------------
# Lorentz force from magnetic field.
# ----------------------------------------------------------------------

def mag(t, y):
    # y[0:3] is position.
    # y[3:6] is velocity
    inv_gamma = np.sqrt(1 - (y[3]**2 + y[4]**2 + y[5]**2)/c**2)
    return np.append(y[3:6], qpm*np.cross(y[3:6], B(t, y[0:3])*inv_gamma))

# ----------------------------------------------------------------------
# Dictionary.
# ----------------------------------------------------------------------

diff_eq_dict = {
    'mag' : mag,  # Lorentz force from magnetic field.
    }  # End of dictionary.

# ======================================================================
# Magnetic field setup functions.
# ======================================================================

def B_maker():  # <=============================================================
    print('Types of magnetic fields available for use:\n'
        + '    H : a homogeneous (constant value) magnetic field.')
    B_setup_dict = {
        'H' : B_H_setup}
    B_key = input('What kind of magnetic field do you want to use?\n>>> ')
    return B_setup_dict[B_key]()

def B_H_setup():
    Bx = input('x component of B field [T]\n>>> ')
    By = input('y component of B field [T]\n>>> ')
    Bz = input('z component of B field [T]\n>>> ')
    return np.asarray([Bx, By, Bz])

# ======================================================================
# Solution approximation method functions and dictionary.
# ======================================================================

# ----------------------------------------------------------------------
# Euler method.
# ----------------------------------------------------------------------

def Eul(t, y, h):
    return (t+h, y+h*f(t, y))

# ----------------------------------------------------------------------
# Second order Runge_Kutta method (RK2).
# ----------------------------------------------------------------------

def RK2(t, y, h):
    hh = 0.5*h
    th = t + h
    k1 = f(t, y)
    k2 = f(t+hh, y+hh*k1)
    return (t+h, y+h*k2)

# ----------------------------------------------------------------------
# Fourth order Runge-Kutta method (RK4, classical Runge–Kutta method).
# ----------------------------------------------------------------------

def RK4(t, y, h):
    hh = 0.5*h
    thh = t + hh
    th = t + h
    k1 = f(t, y)
    k2 = f(thh, y+hh*k1)
    k3 = f(thh, y+hh*k2)
    k4 = f(th, y+h*k3)
    return (th, y+h*(k1+2*(k2+k3)+k4)/6)

# ----------------------------------------------------------------------
# Dictionary.
# ----------------------------------------------------------------------

sol_approx_method_dict = {
    'Eul' : Eul,  # Euler method.
    'RK2' : RK2,  # Second order Runge_Kutta method (RK2).
    'RK4' : RK4,  # Fourth order Runge-Kutta method (RK4).
    }  # End of dictionary.

# ======================================================================
# Initial condition input functions and dictionaries.
# ======================================================================

# ----------------------------------------------------------------------
# Set initial time.
# ----------------------------------------------------------------------

def init_time():
    return float(input('Starting time (t0) [s]:\n>>> '))

# ----------------------------------------------------------------------
# Set unit for angles.
# ----------------------------------------------------------------------

def angel_unit():
    print('Specify the units you will enter angles in:'
        + '\n    rad : radians'
        + '\n    deg : degrees')
    unit = input('Specify the units you will enter angles in:\n>>> ')
    if unit == 'rad':
        factor = 1
    elif unit == 'deg':
        factor = np.pi / 180
    else:
        print('Error in file \'{}\', function \'angle_unit\'.'.format(__file__)
            + '\n    Invalid unit specifier.')
    return unit, factor

# ----------------------------------------------------------------------
# Set initial condition using Cartesian coordinates.
# ----------------------------------------------------------------------

def cart_in(var, unit=None):
    if unit == None:
        print(('Specify starting {} unit vector components (relative entries '
            + 'permitted).').format(var))
        x = float(input('x:\n>>> '))
        y = float(input('y:\n>>> '))
        z = float(input('z:\n>>> '))
        u = np.asarray([x, y, z])
        u /= np.linalg.norm(u)
        return u
    else:
        print(('Specify starting {} vector components.').format(var))
        x = float(input('x [{}]:\n>>> '.format(unit)))
        y = float(input('y [{}]:\n>>> '.format(unit)))
        z = float(input('z [{}]:\n>>> '.format(unit)))
    return np.asarray([x, y, z])

# ----------------------------------------------------------------------
# Set initial condition using cylindrical polar coordinates.
# ----------------------------------------------------------------------

def cyl_in(var, unit=None):
    ang_unit, ang_factor = angle_unit()
    if unit == None:
        print(('Specify starting {} unit vector components (relative entries '
            + 'permitted).').format(var))
        rho = float(input('\u03F1:\n>>> '))
        phi = float(input('\u03C6 [{}]:\n>>> '.format(ang_unit)))*ang_factor
        z = float(input('z:\n>>> '))
        u = np.asarray(cyl_to_cart(rho, phi, z))
        u /= np.linalg.norm(u)
        return u
    else:
        print(('Specify starting {} vector components.').format(var))
        rho = float(input('\u03F1 [{}]:\n>>> '.format(unit)))
        phi = float(input('\u03C6 [{}]:\n>>> '.format(ang_unit)))*ang_factor
        z = float(input('z [{}]:\n>>> '.format(unit)))
    return np.asarray(cyl_to_cart(rho, phi, z))

# ----------------------------------------------------------------------
# Set initial condition using spherical coordinates.
# ----------------------------------------------------------------------

def sph_in(var, unit=None):
    ang_unit, ang_factor = angle_unit()
    if unit == None:
        print(('Specify starting {} unit vector components (relative entries '
            + 'permitted).').format(var))
        phi = float(input('\u03C6 [{}]:\n>>> '.format(ang_unit)))*ang_factor
        theta = float(input('\u03B8 [{}]:\n>>> '.format(ang_unit)))*ang_factor
        u = np.asarray(sph_to_cart(1.0, phi, theta))
        u /= np.linalg.norm(u)
        return u
    else:
        print(('Specify starting {} vector components.').format(var))
        r = float(input('r [{}]:\n>>> '.format(unit)))
        phi = float(input('\u03C6 [{}]:\n>>> '.format(ang_unit)))*ang_factor
        theta = float(input('\u03B8 [{}]:\n>>> '.format(ang_unit)))*ang_factor
    return np.asarray(sph_to_cart(r, phi, theta))

# ----------------------------------------------------------------------
# Coordinate system dictionary.
# ----------------------------------------------------------------------

coord_dict = {
    'cart' : cart_in,
    'cyl'  : cyl_in,
    'sph'  : sph_in}

# ----------------------------------------------------------------------
# Set initial three position.
# ----------------------------------------------------------------------

def init_three_position():
    print('Set initial three-position.')
    coord = input('Available coordinate systems:'
        + '\n    cart : Cartesian coordinates.'
        + '\n    cyl  : cylindrical polar coordinates.'
        + '\n    sph  : spherical coordinates.'
        + '\n>>> ')
    return coord_dict[coord](var='position (r)', unit='m')

# ----------------------------------------------------------------------
# Set initial kinetic energy.
# ----------------------------------------------------------------------

def init_kinetic_energy():
    return input('Select initial kinetic energy [MeV]:\n>>> ')

# ----------------------------------------------------------------------
# Set initial Velocity.
# ----------------------------------------------------------------------

def init_three_velocity():
    T = float(init_kinetic_energy())
    gamma = T/m_MeV + 1.0
    v_norm = c * np.sqrt(1.0 - gamma**-2)
    print('Set initial velocity three-unit-vector.')
    coord = input('Available coordinate systems:'
        + '\n    cart : Cartesian coordinates.'
        + '\n    cyl  : cylindrical polar coordinates.'
        + '\n    sph  : spherical coordinates.'
        + '\n>>> ')
    return coord_dict[coord](var='velocity (v-hat)') * v_norm

# ======================================================================

# ----------------------------------------------------------------------
# Set initial conditions for solution to magnetic field Lorentz force.
# ----------------------------------------------------------------------

def init_cond_mag():
    t0 = init_time()
    print()
    y0 = np.empty(6)
    y0[0:3] = init_three_position()
    print()
    y0[3:6] = init_three_velocity()
    return t0, y0

# ----------------------------------------------------------------------
# Dictionary.
# ----------------------------------------------------------------------

init_cond_setup_dict = {
    'mag' : init_cond_mag}

# ======================================================================
# Iterator function.
# ======================================================================

def setup_iterator(h, t=0.0, y=np.zeros(6), nmax=1000, exitcond=None,
    trackint=None):
    
    # ------------------------------------------------------------------
    # Without tracking.
    # ------------------------------------------------------------------
    if trackint in (0, None):
        
        # --------------------------------------------------------------
        # Without exit condition.
        # --------------------------------------------------------------
        
        if exitcond is None:
            for i in range(nmax):
                t, y = step(t, y, h)
        
        # --------------------------------------------------------------
        # With exit condition.
        # --------------------------------------------------------------
        
        else:
            for i in range(nmax):
                t, y = step(t, y, h)
                if exitcond(t, y, h):
                    break
        
        # --------------------------------------------------------------
        # Return results.
        # --------------------------------------------------------------
        
        return t, y
    
    # ------------------------------------------------------------------
    # With tracking for all steps.
    # ------------------------------------------------------------------
    
    elif (trackint == 1):
        
        # Setup tracking arrays.
        t_arr = np.empty(nmax+1)
        y_arr = np.empty((nmax+1, y.size))
        t_arr[0] = t
        y_arr[0, :] = y
        del t, y
        
        # --------------------------------------------------------------
        # Without exit condition.
        # --------------------------------------------------------------
        
        if exitcond is None:
            for i in range(1, nmax+1):
                t_arr[i], y_arr[i, :] = step(t_arr[i-1], y_arr[i-1, :], h)
        
        # --------------------------------------------------------------
        # With exit condition.
        # --------------------------------------------------------------
        
        else:
            for i in range(1, nmax+1):
                # Update time and position.
                t_arr[i], y_arr[i, :] = step(t_arr[i-1], y_arr[i-1, :], h)
                
                # Terminate loop if exit condition (exitcond) is
                # satisfied and delete excess entries in tracking
                # arrays.
                if exitcond(t_arr[i], y_arr[i, :]):
                    t_arr = np.delete(t_arr, range(i+1, nmax+1), 0)
                    y_arr = np.delete(y_arr, range(i+1, nmax+1), 0)
                    break
        
        # --------------------------------------------------------------
        # Return tracking arrays.
        # --------------------------------------------------------------
        
        return t_arr, y_arr
    
    # ------------------------------------------------------------------
    # With tracking of every trackint-th step.
    # ------------------------------------------------------------------
    
    elif (trackint > 1):
        
        # Number of steps to track.
        ntrack = 1 + (nmax+trackint-1)//trackint
        
        # Setup tracking arrays.
        t_arr = np.empty(ntrack+1)
        y_arr = np.empty((ntrack+1, y.size))
        t_arr[0] = t
        y_arr[0, :] = y
        
        # --------------------------------------------------------------
        # Without exit condition.
        # --------------------------------------------------------------
        
        if exitcond is None:
            for i in range(1, ntrack):
                for j in range(trackint):
                    t, y = step(t, y, h)
                t_arr[i], y_arr[i, :] = t, y
            
            # Calculate final tracked values.  This is separate from
            # main loop to allow for partial intervals when the maximum
            # number of steps (nmax) is not a multiple of tracking
            # interval (trackint).
            for j in range(nmax - (ntrack-2)*trackint):
                t, y = step(t, y, h)
            t_arr[i], y_arr[i, :] = t, y
        
        # --------------------------------------------------------------
        # With exit condition.
        # --------------------------------------------------------------
        
        else:
            for i in range(1, nmax+1):
                for j in range(trackint):
                    t, y = step(t, y, h)
                t_arr[i], y_arr[i, :] = t, y
                
                # Terminate loop if exit condition (exitcond) is
                # satisfied and delete excess entries in tracking
                # arrays.
                if exitcond(t_arr[i], y_arr[i, :]):
                    t_arr = np.delete(t_arr, range(i+1, nmax+1), 0)
                    y_arr = np.delete(y_arr, range(i+1, nmax+1), 0)
                    break
            
            # Calculate final tracked values.  This is separate from
            # main loop to allow for partial intervals when the maximum
            # number of steps (nmax) is not a multiple of tracking
            # interval (trackint).
            else:
                for j in range(nmax - (ntrack-2)*trackint):
                    t, y = step(t, y, h)
                t_arr[i], y_arr[i, :] = t, y
        
        # --------------------------------------------------------------
        # Return tracking arrays.
        # --------------------------------------------------------------
        
        return t_arr, y_arr
    
    # ------------------------------------------------------------------
    # Prints error text if tracking interval (trackint) is not valid.
    else:
        print('Error in file \'{}\', function \'iterator\'.\n'.format(__file__)
            + '    Invalid tracking interval (trackint), trackint must be '
            + 'either a positive\n'
            + '    integer or None.  If trackint is None, the track is not '
            + 'recorded.')
        raise SystemExit

# ======================================================================
# Call main().
# ======================================================================

main()
