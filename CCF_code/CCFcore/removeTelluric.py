import numpy as np

def removeTelluric(wavelens,flux,wmin,wmax):
    """
    Masks the tellurics and reconstructs the Spectrum

    Parameters
    ----------
    wavelens : array
        A wavelength vector for each spectral bin
    flux: array
        The flux corresponding to the wavelength
    wmin : float
        The minimum wavelength at which to start telluric flagging
    wmax : float
        The maximum wavelength at  which to end telluric flagging

    Returns
    -------
    array :
        Tellurics flagged and linearly interpolated array is returned.
    """
    locs_l=list(np.ravel(np.where((wavelens>=np.min(wavelens)) & (wavelens<=wmin))))
    locs_h= list(np.ravel(np.where((wavelens>=wmax) & (wavelens<=np.max(wavelens)))))
    locs=locs_l+locs_h
    flux=np.ravel(flux)
    if (len(locs)==0):
        return flux
    waves=wavelens[locs]
    cut_flux=flux[locs]
    new=np.interp(wavelens,waves,cut_flux)
    return new
