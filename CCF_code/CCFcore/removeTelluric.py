import numpy as np

def removeTelluric(wavelens,flux,wmin,wmax):
    locs_l=list(np.ravel(np.where((wavelens>=np.min(wavelens)) & (wavelens<=wmin))))
    locs_h= list(np.ravel(np.where((wavelens>=wmax) & (wavelens<=np.max(wavelens)))))
    locs=locs_l+locs_h
    flux=np.ravel(flux)
    waves=wavelens[locs]
    cut_flux=flux[locs]
    new=np.interp(wavelens,waves,cut_flux)
    return new

