#!/usr/bin/env python3

"""Converts between different three dimensional coordinate systems.

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

import numpy as np

# Cartesian to cylindrical polar.
def cart_to_cyl(x, y, z):
    return np.sqrt(x**2+y**2), np.arctan2(y, x), z

# Cartesian to spherical.
def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return r, np.arctan2(y, x), np.arccos(z/r)

# Cylindrical polar to Cartesian.
def cyl_to_cart(rho, phi, z):
    return rho*np.cos(phi), rho*np.sin(phi), z

# Cylindrical polar to spherical.
def cyl_to_sph(rho, phi, z):
    r = np.sqrt(rho**2 + z**2)
    return r, phi, np.arccos(z/r)

# Spherical to Cartesian.
def sph_to_cart(r, phi, theta):
    return r*np.sin(theta)*(np.cos(phi), np.sin(phi))

# Spherical to cylindrical polar.
def sph_to_cyl(r, phi, theta):
    return r*sin(theta), phi, r*cos(theta)

# Vectorize functions.
cart_to_cyl = np.vectorize(cart_to_cyl)
cart_to_sph = np.vectorize(cart_to_sph)
cyl_to_cart = np.vectorize(cyl_to_cart)
cyl_to_sph  = np.vectorize(cyl_to_sph)
sph_to_cart = np.vectorize(sph_to_cart)
sph_to_cyl  = np.vectorize(sph_to_cyl)
