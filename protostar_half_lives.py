import numpy as np
import matplotlib.pyplot as plt

def N_0(t, lambda_SF, lambda_0):
    return (lambda_SF / lambda_0) * (1 - np.exp(-lambda_0 * t))

def N_I(t, lambda_SF, lambda_0, lambda_I):
    term1 = (lambda_SF / lambda_I)
    term2 = 1 - (lambda_I / (lambda_I - lambda_0)) * np.exp(-lambda_0 * t)
    term3 = (lambda_0 / (lambda_0 - lambda_I)) * np.exp(-lambda_I * t)
    return term1*(term2-term3)

def N_flat(t, lambda_SF, lambda_0, lambda_I, lambda_flat):
    term1 = (lambda_SF / lambda_flat)
    term2 = (1 - ((lambda_I * lambda_flat) / ((lambda_flat - lambda_0) * (lambda_I - lambda_0))) * np.exp(-lambda_0 * t))
    term3 = (lambda_0 * lambda_flat) / ((lambda_flat - lambda_I) * (lambda_0 - lambda_I)) * np.exp(-lambda_I * t)
    term4 = (lambda_0 * lambda_I) / ((lambda_0 - lambda_flat) * (lambda_I - lambda_flat)) * np.exp(-lambda_flat * t)
    return term1*(term2 - term3 -term4)

def N_II(t, lambda_SF, lambda_0, lambda_I, lambda_flat, lambda_II):
    term1 = (lambda_SF / lambda_II)
    term2 = (1 - ((lambda_I * lambda_flat * lambda_II) / ((lambda_I - lambda_0) * (lambda_flat - lambda_0) * (lambda_II - lambda_0))) * np.exp(-lambda_0 * t))
    term3 = (lambda_0 * lambda_flat * lambda_II) / ((lambda_0 - lambda_I) * (lambda_flat - lambda_I) * (lambda_II - lambda_I)) * np.exp(-lambda_I * t)
    term4 = (lambda_0 * lambda_I * lambda_II) / ((lambda_0 - lambda_flat) * (lambda_I - lambda_flat) * (lambda_II - lambda_flat)) * np.exp(-lambda_flat * t)
    term5 = (lambda_0 * lambda_I * lambda_flat) / ((lambda_0 - lambda_II) * (lambda_I - lambda_II) * (lambda_flat - lambda_II)) * np.exp(-lambda_II * t)
    return term1*( term2 - term3 - term4 - term5)

def plot_star_formation_numbers():

    plt.plot(t,N0_values,label='N0',lw=1)
    plt.plot(t,NI_values,label='NI',lw=2)
    plt.plot(t,Nflat_values,label='NFS',lw=3)
    plt.plot(t,NII_values,label='NII')
    plt.plot(t,NIII_values,label='NIII')
    plt.plot(t,Ntot,label='Ntot',color='k')

    plt.legend()
    plt.yscale("log")
    plt.ylim(1e0,5e3)
    plt.show()

def plot_star_formation_fraction():

    plt.plot(t,N0_values/Ntot,label='N0',lw=1)
    plt.plot(t,NI_values/Ntot,label='NI',lw=2)
    plt.plot(t,Nflat_values/Ntot,label='NFS',lw=3)
    plt.plot(t,NII_values/Ntot,label='NII')
    plt.plot(t,NIII_values/Ntot,label='NIII')
    # plt.plot(t,Ntot,label='Ntot',color='k')

    plt.legend()
    # plt.yscale("log")
    # plt.ylim(1e0,5e3)
    plt.show()
def plot_star_formation_relative_fraction():

    plt.plot(t,Nflat_values/NI_values,label='Nflat/NI',lw=2)
    plt.plot(t,NII_values/Nflat_values,label='NII/NFS',lw=3)
    plt.plot(t,NII_values/NI_values,label='NII/NI',lw=2)

    # plt.plot(t,Ntot,label='Ntot',color='k')
    plt.legend()
    # plt.yscale("log")
    # plt.ylim(1e0,5e3)
    plt.show()

if __name__ == "__main__":
    t = np.linspace(1e-2, 5.0, 300)  # Time from 0 to 10 in 100 steps
    lambda_SF = 830
    lambda_0 = 14.7
    lambda_I = 7.9
    lambda_flat = 8.0
    lambda_II = 0.347 ### ln(2)/2Myr = 0.347 Myr

    # Calculating the values
    N0_values = N_0(t, lambda_SF, lambda_0)
    NI_values = N_I(t, lambda_SF, lambda_0, lambda_I)
    Nflat_values = N_flat(t, lambda_SF, lambda_0, lambda_I, lambda_flat)
    NII_values = N_II(t, lambda_SF, lambda_0, lambda_I, lambda_flat, lambda_II)
    Ntot = lambda_SF*t
    NIII_values = Ntot - N0_values - NI_values - Nflat_values - NII_values
    # plot_star_formation_numbers()
    # plot_star_formation_fraction()
    plot_star_formation_relative_fraction()
