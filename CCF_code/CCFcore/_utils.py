
# coding: utf-8

# # HD206893: data reduction

# *Version 1 (2017/08)* 
# 
# *Author: Valentin Christiaens*


import numpy as np
import pdb
from vip_hci.fits import open_fits

# Definition of Useful functions

def find_nearest(array, value, output='index', constraint=None):
    """
    Function to find the index, and optionally the value, of an array's closest element to a certain value.
    Possible outputs: 'index','value','both' 
    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest element with a value greater than 'value', "floor" the opposite)
    """
    if type(array) is np.ndarray:
        pass
    elif type(array) is list:
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")
        
    idx = (np.abs(array-value)).argmin()
    if type == 'ceil' and array[idx]-value < 0:
        idx+=1
    elif type == 'floor' and value-array[idx] < 0:
        idx-=1

    if output=='index': return idx
    elif output=='value': return array[idx]
    else: return array[idx], idx


def find_derot_angles(inpath, fits_list, loc='st', ipag=True, verbose=False):
    """ 
    Find the derotation angle vector to apply to a set of NACO cubes to align it with North up.
    IMPORTANT: the list of fits has to be in chronological order of acquisition.
    
    Parameters:
    ***********
    
    inpath: str
        Where the fits files are located
    fits_list: list of str
        List containing the name of every fits file to be considered
    loc: {'st','nd'}, str, opt
        Whether to consider the derotation angle at the beginning ('st') or end 
        ('nd') of the cube exposure.
    ipag: {False, True}, bool, opt
        Whether to use the method recommended by ipag. If not, just return the value
        of keyword 'POSANG'
    """
    
    n_fits = len(fits_list)
    rot_pt_off = np.zeros(n_fits)
    parang = np.zeros(n_fits)
    final_derot_angs = np.zeros(n_fits)
    posang = np.zeros(n_fits)
    
    if loc == 'st':
        kw_par = 'HIERARCH ESO TEL PARANG START'
        kw_pos = 'HIERARCH ESO ADA POSANG'
    elif loc == 'nd':
        kw_par = 'HIERARCH ESO TEL PARANG END'
        kw_pos = 'HIERARCH ESO ADA POSANG END'
  
    # FIRST COMPILE PARANG, POSANG and PUPILPOS 
    for ff in range(n_fits):
        _, header = open_fits(inpath+fits_list[ff], header=True, verbose=False)
        parang[ff] = header[kw_par]
        posang[ff] = header[kw_pos]
        pupilpos = 180.0 - parang[ff] + posang[ff]
        rot_pt_off[ff] = 90 + 89.44 - pupilpos
        if verbose:
            print("parang: {}, posang: {}, rot_pt_off: {}".format(parang[ff],posang[ff],rot_pt_off[ff]))
       
    # NEXT CHECK IF THE OBSERVATION WENT THROUGH TRANSIT (change of sign in parang OR stddev of rot_pt_off > 1.)       
       
    rot_pt_off_med = np.median(rot_pt_off)
    rot_pt_off_std = np.std(rot_pt_off)    
    
    if np.min(parang)*np.max(parang) < 0. or rot_pt_off_std > 1.:
        if verbose:
            print("The observation goes through transit and/or the pupil position was reset in the middle of the observation: ")
            if np.min(parang)*np.max(parang) < 0.:
                print("min/max parang: ", np.min(parang), np.max(parang))
            if rot_pt_off_std > 1.:
                print("the standard deviation of pupil positions is greater than 1: ", rot_pt_off_std)
        # find index where the transit occurs (change of sign of parang OR big difference in pupil pos)
        n_changes = 0
        for ff in range(n_fits-1):
            if parang[ff]*parang[ff+1] < 0. or np.abs(rot_pt_off[ff]-rot_pt_off[ff+1]) > 1.:
                idx_transit = ff+1
                n_changes+=1
        # check that these conditions only detected one passage through transit
        if n_changes != 1:
            print(" {} passages of transit were detected (instead of 1!). Check that the input fits list is given in chronological order.".format(n_changes))
            pdb.set_trace()
    
        rot_pt_off_med1 = np.median(rot_pt_off[:idx_transit])    
        rot_pt_off_med2 = np.median(rot_pt_off[idx_transit:])
        
        final_derot_angs = rot_pt_off_med1 - parang
        final_derot_angs[idx_transit:] = rot_pt_off_med2 - parang[idx_transit:]
    
    else:
        final_derot_angs = rot_pt_off_med - parang

    # MAKE SURE ANGLES ARE IN THE RANGE (-180,180)deg
    min_derot_angs = np.amin(final_derot_angs)
    nrot_min = min_derot_angs/360.
    if nrot_min < -0.5:
        final_derot_angs[np.where(final_derot_angs<-180)] = final_derot_angs[np.where(final_derot_angs<-180)] + np.ceil(nrot_min)*360.
    max_derot_angs = np.amax(final_derot_angs)
    nrot_max = max_derot_angs/360.
    if nrot_max > 0.5:
        final_derot_angs[np.where(final_derot_angs>180)] = final_derot_angs[np.where(final_derot_angs>180)] - np.ceil(nrot_max)*360.
        
    if ipag:
        return -1.*final_derot_angs
    else:
        return posang
def maxMinNorm(vector):
    vector=np.asarray(vector,dtype=np.float64)
    return (vector-np.nanmin(vector))/(np.nanmax(vector)-np.nanmin(vector))