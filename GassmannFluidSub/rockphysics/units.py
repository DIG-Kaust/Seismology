import numpy as np

def kg_m3_to_g_cm3(x):
    """convert from kg/m^3 to g/cm^3
    """
    return x * 1e-3

def g_cm3_to_kg_m3(x):
    """convert from g/cm^3 to kg/m^3
    """
    return x * 1e3

def psi_to_Pa(x):
    """convert from psi to Pa
    """
    return x * 6.894757

def Pa_to_psi(x):
    """convert from Pa to psi
    """
    return x / 6.894757


def ft_to_m(x):
    """convert from feet to m
    """
    return x * 0.305

def m_to_ft(x):
    """convert from m to feet
    """
    return x / 0.305