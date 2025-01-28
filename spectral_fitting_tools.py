def gauss(x, amp, mean, sigma):
    g = amp*np.exp(-(((x-mean)**2)/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))
    return g

def gauss_2line(x, *pars):
    amp2, z, sigma_v, cont, amp3 = pars
    amp1 = amp3/3

    wl1 = NIIa_rest * (1+z)
    wl2 = Ha_rest * (1+z)
    wl3 = NIIb_rest * (1+z)

    wla = dtools.vtowl(-sigma_v/2,wl1) #wl of start of FWHM
    wlb = dtools.vtowl(sigma_v/2,wl1) #wl of end of FWHM
    sigma_wl = abs(wlb - wla)

    g1 = gauss(x, amp1, wl1, sigma_wl) #Gauss line1
    g2 = gauss(x, amp2, wl2, sigma_wl) 
    g3 = gauss(x, amp3, wl3, sigma_wl) 
    return cont + g1 + g2 + g3

