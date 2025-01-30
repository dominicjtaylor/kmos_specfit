import numpy as np
import dtools
from scipy.optimize import curve_fit

cat_data = dtools.get_IRlines()
Ha_rest = cat_data['Ha']
NIIa_rest = cat_data['NII_a']
NIIb_rest = cat_data['NII_b']
Hb_rest = cat_data['Hb']
OIIIa_rest = cat_data['OIII_a']
OIIIb_rest = cat_data['OIII_b']

class specfit:
    @staticmethod
    def gauss(x, amp, mean, sigma):
        """
        Normalised Gaussian function so that the amplitude is equal to the flux
        """
        return amp*np.exp(-(((x-mean)**2)/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))

    @staticmethod
    def gauss_Ha_NIIab(x, *pars):
        """
        Gaussian profile for H-alpha and [NII] doublet.
        Made by defining a sigma in velocity
        Includes a continuum fit
        ---------------
        x: wavelength list
        *pars: optimal parameters (tup)
        ---------------
        """
        amp2, z, sigma_v, cont, amp3 = pars
        amp1 = amp3 / 3

        # Calculate shifted wavelengths for each line
        wl1 = NIIa_rest * (1+z)
        wl2 = Ha_rest * (1+z)
        wl3 = NIIb_rest * (1+z)

        # Convert velocity dispersion to wavelength dispersion
        wla = dtools.vtowl(-sigma_v/2, wl1)  # Start of FWHM
        wlb = dtools.vtowl(sigma_v/2, wl1)   # End of FWHM
        sigma_wl = abs(wlb - wla)

        # Create Gaussian profiles for each line
        g1 = specfit.gauss(x, amp1, wl1, sigma_wl)
        g2 = specfit.gauss(x, amp2, wl2, sigma_wl)
        g3 = specfit.gauss(x, amp3, wl3, sigma_wl)

        return g1 + g2 + g3 + cont

    @staticmethod
    def fit_Ha_NIIab(wl,spec,z0,p0_sigma=200,p0_cont=20,dl=0.001,atmos_abs1=1.8,atmos_abs2=1.95):
        """
        Function to fit gaussian profiles to data, using initial guesses
        ---------------
        wl: wavelength list
        spec: flux density list (same length as wl)
        z0: initial guess redshift
        p0_sigma: initial guess line sigma in velocity
        p0_cont: initial guess amplitude of the continuum
        dl: wavelength window around each line to guess its amplitude
        atmos_abs1: start of wavelength range defining atmospheric absorption
        atmos_abs2: end of wavelength range defining atmospheric absorption
        ---------------
        """

        wl = np.array(wl)
        spec = np.array(spec)

        #Apply weight of 0 to wavelength range subject to sky absorption
        weights = np.ones(len(wl))
        sky_mask = (wl > atmos_abs1) & (wl < atmos_abs2) #for HK band
        weights[sky_mask] = 1e6
        sigma = 1/weights
        # sky += abs(min(sky))+1e-5
        # sky /= min(sky)
        # sigma = sky #error

        #Initial guesses
        wl_Ha = Ha_rest * (1+z0)
        wl_NIIb = NIIb_rest * (1+z0)
        Ha_mask = (wl > wl_Ha - dl/2) & (wl < wl_Ha + dl/2)
        NIIb_mask = (wl > wl_NIIb - dl/2) & (wl < wl_NIIb + dl/2)
        amp0_Ha = max(spec[Ha_mask]) / p0_sigma*np.sqrt(2*np.pi)
        amp0_NIIb = max(spec[NIIb_mask]) / p0_sigma*np.sqrt(2*np.pi)
        p0 = [amp0_Ha,z0,p0_sigma,p0_cont,amp0_NIIb] #best amp1/2, mean1/2, sigma1, sigma2

        #Bounds
        bounds = [(0,z0-0.01,0,0,0),
                  (np.inf,z0+0.01,np.inf,np.inf,np.inf)]

        #Fit
        popt, pcov = curve_fit(specfit.gauss_Ha_NIIab,wl,spec,p0,sigma=sigma,bounds=bounds,maxfev=500000)
        gfit = specfit.gauss_Ha_NIIab(wl,*popt)

        return gfit, popt

    @staticmethod
    def gauss_Hb_OIIIab(x,*pars):
        """
        Gaussian profile for H-beta and [OIII] doublet.
        Made by defining a sigma in velocity
        Includes a continuum fit
        ---------------
        x: wavelength list
        *pars: optimal parameters (tup)
        ---------------
        """
        amp1, z, sigma_v, cont, amp2, amp3 = pars

        # Calculate shifted wavelengths for each line
        wl1 = OIIIa_rest * (1+z)
        wl2 = Hb_rest * (1+z)
        wl3 = OIIIb_rest * (1+z)

        # Convert velocity dispersion to wavelength dispersion
        wla = dtools.vtowl(-sigma_v/2,wl1) #wl of start of FWHM
        wlb = dtools.vtowl(sigma_v/2,wl1) #wl of end of FWHM
        sigma_wl = abs(wlb - wla)

        # Create Gaussian profiles for each line
        g1 = specfit.gauss(x, amp1, wl1, sigma_wl) #Gauss line1
        g2 = specfit.gauss(x, amp2, wl2, sigma_wl)
        g3 = specfit.gauss(x, amp3, wl3, sigma_wl)

        return cont + g1 + g2 + g3

    @staticmethod
    def fit_Hb_OIIIab(wl,spec,z0,p0_sigma=200,p0_cont=20,dl=0.002,atmos_abs1=1.8,atmos_abs2=1.95):
        """
        Function to fit gaussian profiles to data, using initial guesses
        ---------------
        wl: wavelength list
        spec: flux density list (same length as wl)
        z0: initial guess redshift
        p0_sigma: initial guess line sigma in velocity
        p0_cont: initial guess amplitude of the continuum
        dl: wavelength window around each line to guess its amplitude
        atmos_abs1: start of wavelength range defining atmospheric absorption
        atmos_abs2: end of wavelength range defining atmospheric absorption
        ---------------
        """

        wl = np.array(wl)
        spec = np.array(spec)

        weights = np.ones(len(wl))
        sky_mask = (wl > atmos_abs1) & (wl < atmos_abs2)
        weights[sky_mask] = 1e6
        sigma = 1/weights

        #Initial guesses
        wl_OIIIa = OIIIa_rest * (1+z0)
        wl_Hb = Hb_rest * (1+z0)
        wl_OIIIb = OIIIb_rest * (1+z0)
        OIIIa_mask = (wl > wl_OIIIa - dl/2) & (wl < wl_OIIIa + dl/2)
        Hb_mask = (wl > wl_Hb - dl/2) & (wl < wl_Hb + dl/2)
        OIIIb_mask = (wl > wl_OIIIb - dl/2) & (wl < wl_OIIIb + dl/2)
        amp0_OIIIa = max(spec[OIIIa_mask]) / p0_sigma*np.sqrt(2*np.pi)
        amp0_Hb = max(spec[Hb_mask]) / p0_sigma*np.sqrt(2*np.pi)
        amp0_OIIIb = max(spec[OIIIb_mask]) / p0_sigma*np.sqrt(2*np.pi)
        p0 = [amp0_OIIIa,z0,p0_sigma,p0_cont,amp0_Hb,amp0_OIIIb] #best amp1/2, mean1/2, sigma1, sigma2

        #Bounds
        bounds = [(0,z0-0.01,0,0,0,0),
                  (np.inf,z0+0.01,np.inf,np.inf,np.inf,np.inf)]

        #Fit
        popt, pcov = curve_fit(specfit.gauss_Hb_OIIIab,wl,spec,p0,sigma=sigma,bounds=bounds,maxfev=500000)
        gfit = specfit.gauss_Hb_OIIIab(wl,*popt)

        return gfit, popt

