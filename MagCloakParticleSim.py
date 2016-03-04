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

from coordinate_converter import *

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
    
    # Differential equation. ===========================================
    
    print('\n'
        + 'Available differential equations.\n'
        + '    mag : Lorentz force from magnetic field.')
    
    diff_eq_key = input(
        'Which differential equation do you want to use?\n>>> ')
    print()
    f = diff_eq_dict[diff_eq_key]
    
    # Fields required by differential equation. ========================
    
    # Magnetic field.
    if diff_eq_key in ('mag'):
        B = B_maker()
    print()
    
    # Solution approximation method. ===================================
    
    step_key = input(
        'Which solution approximation method do you want to use?\n>>> ')
    print()
    step = sol_approx_method_dict[step_key]
    
    # Tracking settings. ===============================================
    
    trackint = int(input(
        'What tracking interval would you like to use?\n>>> '))
    print()
    
    # Maximum number of iterations. ====================================
    
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
    
    # Iteration exit conditions. =======================================
    
    print('Available exit conditions:'
        + '\n    0 : none, will run nmax iterations.'
        #+ '\n    box : a box of user specified dimensions.'
        )
    exitcond_key = input('Which exit condition would you like to use?\n>>> ')
    print()
    
    # Initial conditions. ==============================================
    
    t0, y0 = init_cond_setup_dict[diff_eq_key]()
    print()
    
    # Step size. =======================================================
    
    print('Ways of specifying step size available:'
        + '\n    dt : a time interval.'
        #+ '\n    dl : a distance (converted to time based on initial speed).'
        )
    h_type = input('How would you like to specify step size?\n>>> ')
    h_type_dict[h_type]
    print()
    
    # ==================================================================
    # Run simulation.
    # ==================================================================
    
    iterator(
#        step,
        h,
        #t=t0,
        #y=y0,
        nmax=nmax,
        #exitcond=exitcond,
        #trackint=trackint
        )
    
    # ==================================================================
    # End of function.
    # ==================================================================
    
    return 0

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

# Lorentz force from magnetic field. ===================================
def mag(t, y):
    # y[0:3] is position.
    # y[3:6] is velocity
    return np.append(y[3:6], qgm*np.cross(y[3:6], B(t, y[0:3])))

# Dictionary. ==========================================================
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
    Bx = input('x component of B field\n>>> ')
    By = input('y component of B field\n>>> ')
    Bz = input('z component of B field\n>>> ')
    return np.asarray([Bx, By, Bz])

# ======================================================================
# Solution approximation method functions and dictionary.
# ======================================================================

# Euler method. ========================================================
def Eul(t, y, h):
    return (t+h, y+h*f(t, y))

# Second order Runge_Kutta method (RK2). ===============================
def RK2(t, y, h):
    hh = 0.5*h
    th = t + h
    k1 = f(t, y)
    k2 = f(t+hh, y+hh*k1)
    return (t+h, y+h*k2)

# Fourth order Runge-Kutta method (RK4, classical Runge–Kutta method). =
def RK4(t, y, h):
    hh = 0.5*h
    thh = t + hh
    th = t + h
    k1 = f(t, y)
    k2 = f(thh, y+hh*k1)
    k3 = f(thh, y+hh*k2)
    k4 = f(th, y+h*k3)
    return (th, y+h*(k1+2*(k2+k3)+k4)/6)

# Dictionary. ==========================================================
sol_approx_method_dict = {
    'Eul' : Eul,  # Euler method.
    'RK2' : RK2,  # Second order Runge_Kutta method (RK2).
    'RK4' : RK4,  # Fourth order Runge-Kutta method (RK4).
    }  # End of dictionary.

# ======================================================================
# Initial condition input functions and dictionaries.
# ======================================================================

def t0():
    return float(input('Starting time (t0) [s]:\n>>> '))

def cart_in(var, units=None, unit_vect=False):
    if unit_vect:
        print(('Specify starting {} unit vector components (relative entries '
            + 'permitted).').format(var))
        x = float(input('x:\n>>> '))
        y = float(input('y:\n>>> '))
        z = float(input('z:\n>>> '))
        u = np.asarray([x, y, z])
        u /= np.linalg.norm(v)
        return u
    else:
        print(('Specify starting {} vector components.').format(var))
        x = float(input('x [{}]:\n>>> '.format(units)))
        y = float(input('y [{}]:\n>>> '.format(units)))
        z = float(input('z [{}]:\n>>> '.format(units)))
    return np.asarray([x, y, z])

def cyl_in(var, units=None, unit_vect=False):
    if unit_vect:
        print(('Specify starting {} unit vector components (relative entries '
            + 'permitted).').format(var))
        rho = float(input('\u03F1:\n>>> '))
        phi = float(input('\u03C6 [{}]:\n>>> '.format(units[1])))
        z = float(input('z:\n>>> '))
        u = np.asarray(cyl_to_cart(rho, phi, z))
        u /= np.linalg.norm(u)
        return u
    else:
        print(('Specify starting {} vector components.').format(var))
        rho = float(input('\u03F1 [{}]:\n>>> '.format(units)))
        phi = float(input('\u03C6 [{}]:\n>>> '.format(units)))
        z = float(input('z [{}]:\n>>> '.format(units)))
    return np.asarray(cyl_to_cart(rho, phi, z))

coord_dict = {
    'cart' : cart_in,
    'cyl'  : cyl_in,
    #'sph'  : sph_in
    }

def three_position():
    coord = input('Available coordinate systems:'
        + '\n    cart : Cartesian coordinates.'
        + '\n    cyl  : cylindrical polar coordinates.'
        + '\n    sph  : spherical coordinates.'
        + '\n>>> ')
    
    
    return

# Dictionary. ==========================================================
init_cond_setup_dict = {
    'mag' : (t0, three_position, )}


# ======================================================================
# Iterator function.
# ======================================================================

#def iterator(step, h, t=0.0, y=np.zeros(6), nmax=1000, exitcond=None,
#  trackint=None):
def iterator(h, t=0.0, y=np.zeros(6), nmax=1000, exitcond=None, trackint=None):
    '''Iterates the updating of parameters of a differential equation.
    
    Parameters
    ----------
    t : float
        The starting time.
    y : numpy.ndarray
        The initial value for the parameters of f other than time.
    nmax : int, optional
        The maximum number of steps to iterate over.
    exitcond : function or None, optional
        The exit condition for the iteration.  If none is given, the
        iteration will terminate after nmax iterations.
    trackint : int or None, optional
        The tracking interval.  The number of steps the iterator should
        calculate between storing the results to the tracking arrays
        (i.e. the iterator will save store the result of every
        trackint-th step).  If trackint is None, tracking is disabled. 
    '''
    
    # If tracking is disabled (trackint is None).
    if trackint in (0, None):
        
        # If there is no exit condition (exitcond is None).
        if exitcond is None:
            for i in range(nmax):
                t, y = step(t, y, h)
        
        # If there is an exit condition (exitcond is not None).
        else:
            for i in range(nmax):
                t, y = step(t, y, h)
                if exitcond(t, y, h):
                    break
        
        return t, y
    
    # If tracking is enabled for all steps.
    elif (trackint == 1):
        
        # Setup tracking arrays.
        t_arr = np.empty(nmax+1)
        y_arr = np.empty((nmax+1, y.size))
        t_arr[0] = t
        y_arr[0, :] = y
        del t, y
        
        # If there is no exit condition (exitcond is None).
        if exitcond is None:
            for i in range(1, nmax+1):
                t_arr[i], y_arr[i, :] = step(t_arr[i-1], y_arr[i-1, :], h)
        
        # If there is an exit condition (exitcond is not None).
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
        return t_arr, y_arr
    
    # If tracking interval (trackint) is greater than one, takes
    # trackint many steps between recording values to tracking arrays.
    elif (trackint > 1):
        
        # Number of steps to track.
        ntrack = 1 + (nmax+trackint-1)//trackint
        
        # Setup tracking arrays.
        t_arr = np.empty(ntrack+1)
        y_arr = np.empty((ntrack+1, y.size))
        t_arr[0] = t
        y_arr[0, :] = y
        
        # If there is no exit condition (exitcond is None).
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
        
        # If there is an exit condition (exitcond is not None).
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
        
        return t_arr, y_arr
    
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
