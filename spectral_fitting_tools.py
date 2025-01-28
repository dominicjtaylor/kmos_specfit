def gauss(x, amp, mean, sigma):
    g = amp*np.exp(-(((x-mean)**2)/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))
    return g

def gauss_Ha_NIIab(x, *pars):
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

def fit_Ha_NIIab(wl,spec,z0,sky):
    weights = wl*0 + 1
    sky_mask = (wl > 1.8) & (wl < 1.95) #for HK band
    weights[sky_mask] = 1e6
    sigma = 1/weights
    # sky += abs(min(sky))+1e-5
    # sky /= min(sky)
    # sigma = sky #error

    #Initial guesses
    p0_sigma = 200
    p0_cont = 20
    wl_Ha = Ha_rest * (1+z0)
    wl_NIIb = NIIb_rest * (1+z0)
    dl = 0.001
    Ha_mask = (wl > wl_Ha - dl) & (wl < wl_Ha + dl)
    NIIb_mask = (wl > wl_NIIb - dl) & (wl < wl_NIIb + dl)
    amp0_Ha = max(spec[Ha_mask]) / p0_sigma*np.sqrt(2*np.pi)
    amp0_NIIb = max(spec[NIIb_mask]) / p0_sigma*np.sqrt(2*np.pi)
    p0 = [amp0_Ha,z0,p0_sigma,p0_cont,amp0_NIIb] #best amp1/2, mean1/2, sigma1, sigma2
    bounds = [(0,z0-0.01,0,0,0),
              (np.inf,z0+0.01,np.inf,np.inf,np.inf)]

    #Fit
    popt, pcov = curve_fit(gauss_Ha_NIIab,wl,spec,p0,sigma=sigma,bounds=bounds,maxfev=500000)
    gfit = gauss_3line(wl,*popt)

    return gfit, popt

def gauss_Hb_OIIIab(x, *pars):
    amp1, z, sigma_v, cont, amp2, amp3 = pars

    wl1 = OIIIa_rest * (1+z)
    wl2 = Hb_rest * (1+z)
    wl3 = OIIIb_rest * (1+z)

    wla = dtools.vtowl(-sigma_v/2,wl1) #wl of start of FWHM
    wlb = dtools.vtowl(sigma_v/2,wl1) #wl of end of FWHM
    sigma_wl = abs(wlb - wla)

    g1 = gauss(x, amp1, wl1, sigma_wl) #Gauss line1
    g2 = gauss(x, amp2, wl2, sigma_wl)
    g3 = gauss(x, amp3, wl3, sigma_wl)
    return cont + g1 + g2 + g3

def fit_Hb_OIIIab(wl,spec,z0,sky):
    wl = np.array(wl)
    spec = np.array(spec)
    sky = np.array(sky)

    weights = np.ones(len(wl))
    sky_mask = (wl > 1.8) & (wl < 1.95)
    weights[sky_mask] = 1e6
    sigma = 1/weights

    #Initial guesses
    p0_sigma = 200
    p0_cont = 20
    wl_OIIIa = OIIIa_rest * (1+z0)
    wl_Hb = Hb_rest * (1+z0)
    wl_OIIIb = OIIIb_rest * (1+z0)
    dl = 0.001
    OIIIa_mask = (wl > wl_OIIIa - dl) & (wl < wl_OIIIa + dl)
    Hb_mask = (wl > wl_Hb - dl) & (wl < wl_Hb + dl)
    OIIIb_mask = (wl > wl_OIIIb - dl) & (wl < wl_OIIIb + dl)
    amp0_OIIIa = max(spec[OIIIa_mask]) / p0_sigma*np.sqrt(2*np.pi)
    amp0_Hb = max(spec[Hb_mask]) / p0_sigma*np.sqrt(2*np.pi)
    amp0_OIIIb = max(spec[OIIIb_mask]) / p0_sigma*np.sqrt(2*np.pi)
    p0 = [amp0_OIIIa,z0,p0_sigma,p0_cont,amp0_Hb,amp0_OIIIb] #best amp1/2, mean1/2, sigma1, sigma2
    bounds = [(0,z0-0.01,0,0,0,0),
    ¦   ¦   ¦ (np.inf,z0+0.01,np.inf,np.inf,np.inf,np.inf)]

    #Fit
    popt, pcov = curve_fit(gauss_Hb_OIIIab,wl,spec,p0,sigma=sigma,bounds=bounds,maxfev=500000)
    gfit = gauss_Hb_OIIIab(wl,*popt)

    return gfit, popt

