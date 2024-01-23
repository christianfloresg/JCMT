import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy.ma as ma

# import astropy.constants as c
# cc = c.c.cgs.value

def read_text_file(filename):

    source_name, logg,logg_err, temp, hco_c, author,binary = [],[],[],[],[],[],[]
    with open(filename) as f:
        for line in f:
            array = line.strip().split()
            if '#' not in array[0]:
                source_name.append(array[0])
                logg.append(array[1])
                logg_err.append(array[2])
                temp.append(array[3])
                hco_c.append(array[4])
                author.append(array[5])
                binary.append(array[6])
                # print(array)

    logg_float = np.array(logg, dtype=float)
    logg_err_float = np.array([[float(value.split(',')[0]),float(value.split(',')[1])] for value in logg_err])
    temp_float = np.array(temp, dtype=float)

    hco_float = []

    for value in hco_c:
        try:
            hco_float.append(float(value))
        except ValueError:
            hco_float.append(0.2)

    return logg_float,logg_err_float,temp_float,hco_float,source_name,author,binary

def plot_envelope_disspation(filename):


    logg_float,logg_err_float, temp_float, hco_float, source_name, author,binary = read_text_file(filename)
    logg_err=0.01
    hco_err=0.01
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)

    '''
    plot data
    '''

    ###Author masks
    # our_data_mask = np.invert([item.lower() == 'me' for item in author])
    # carney_data_mask = np.invert([item.lower() == 'c' for item in author])
    # vanKepen_data_mask = np.invert([item.lower() == 'vk' for item in author])
    # legend=['me','C','vK']
    # mask_array = [our_data_mask,carney_data_mask,vanKepen_data_mask]


    #### temperature mask
    # lower_mask=ma.masked_greater_equal(temp_float,3400).mask
    # upper_mask=ma.masked_less(temp_float,3400).mask
    # mask_array = [lower_mask,upper_mask]
    # legend=['cold','hot']
    #
    #
    ### binary mask
    single_mask = np.invert([item.lower() == 's' for item in binary])
    binary_mask = np.invert([item.lower() == 'cb' for item in binary])
    multiple_mask = np.invert([item.lower() == 'hm' or item.lower() == 'wb' for item in binary])
    legend=['s','b','m']
    mask_array = [single_mask,binary_mask,multiple_mask]


    symbol_size = 16

    colors=['C0','C1','C2']
    for ii in range(len(mask_array)):
        double_mask = [[element, element] for element in mask_array[ii]]
        # print(double_mask)

        logg_masked = ma.masked_array(logg_float,mask_array[ii]).compressed()
        logg_err_masked = ma.masked_array(logg_err_float,double_mask).compressed()
        hco_masked = ma.masked_array(hco_float,mask_array[ii]).compressed()
        name_masked = ma.masked_array(source_name,mask_array[ii]).compressed()

        a = [logg_err_masked[1::2],logg_err_masked[::2]] #np.reshape(logg_err_masked, (2, int(len(logg_err_masked)/2)))
        ax.errorbar(logg_masked,hco_masked, yerr=hco_err,xerr=a,linestyle='none',
                    ecolor='gray',marker='o',markerfacecolor=colors[ii],ms=symbol_size,mec='k',mew=1.5,capsize=3
                    ,label=legend[ii])


    '''
    manage axis
    '''

    #ax.axvline(x=3.35,color='k',linestyle='--')
    font_size = 14

    ax.set_xlabel(r'$\log{g}$'+ ' [stellar gravity]', fontsize=font_size+4)
    ax.set_ylabel(r'$C_{HCO^+}$' +' [concentration factor]' , fontsize=font_size+4)
    ax.tick_params(direction="in",which='both', labelsize=16)
    ax.set_xlim(2.20,4.1)
    ax.set_ylim(0.1,0.91)

    ax2 = ax.twiny()
    ax2.set_xlabel('Age (Myr)',fontsize=font_size+4)

    newlabel = [ ' ', 0.1, 0.2, 0.4, 1.3, 2.3]
    newpos = np.linspace(0.1,0.9,6)  # position of the xticklabels in the old x-axis
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(newlabel)
    ax2.tick_params(direction="in",which='both', labelsize=16)
    ax.legend()
    plt.savefig(fig_name, bbox_inches='tight', dpi=200)
    plt.show()


if __name__ == "__main__":
    file='concentration_factors.txt'
    fig_name='Envelope_by_binary.pdf'
    plot_envelope_disspation(file)
    # read_text_file('concentration_factors.txt')
    # read_text_file(file)
