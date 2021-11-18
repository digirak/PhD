from matplotlib import pyplot as plt
import numpy as np
import glob
from vip_hci.fits import open_fits, write_fits
from scipy.signal import convolve2d
import pandas as pd
plt.style.use("seaborn-poster")
import matplotlib as mpl
import warnings
from astropy.table import Table
font = {'family' : 'serif',
        'weight' : 'bold'}

mpl.rc('font', **font)

files=glob.glob("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/Telluricsadded/TelluricsIncl/TelluricsIncl/SNR/*")

def create_circle(loc,r,color,fill=False,label=None):
	circle= plt.Circle(loc, radius= r,fill=fill,color=color,lw=2.0,label=label)
	return circle

filenames=[]
for fi in files:
    filenames.append(fi.split('/')[-1][-55::])
t1=Table.read("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/astrometry_new_psf.fits")
p=t1.to_pandas()

#p=p.drop_duplicates()

p.loc[((p['wmin']==1.9)& (p['wmax']==2.5)),'band']='K'
p.loc[((p['wmin']==1.9)& (p['wmax']==2.3)),'band']='K (1.9-2.3)'
p.loc[((p['wmin']==2.0)& (p['wmax']==2.4)),'band']='K (2.0-2.4)'

p.loc[((p['wmin']==1.4) &(p['wmax']==2.5)),'band']='Whole'
p.loc[((p['wmin']>=1.4) &(p['wmax']<=1.9)),'band']='H'
p.loc[((p['wmin']>=1.4) &(p['wmax']<1.9)),'band']='H short'
p=p.drop_duplicates()
p.dropna(inplace=True)
psf=open_fits("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/normalize_psf.fits")
    


for i in range(len(filenames)):
    data=open_fits(files[i])
    f=filenames[i]                    
    #f=files[i].split('/')[-1]
    PCs=int(f.split('_')[3])
    wmin=np.float64(f.split('_')[5])
    wmax=np.float64(f.split('_')[7])
    print("Computing the values between {0} and {1} for {2} PCs".format(wmin,wmax,PCs))
    

    snrplot=convolve2d(data,psf,mode='same')

    filt=p.loc[(p['wmin']==wmin) &( p['wmax']==wmax) &(p['PCs']==PCs)]
    xx_v=filt['xx_valentin'].values
    yy_v=filt['yy_valentin'].values
    xx=filt['xx_rakesh_psf'].values
    yy=filt['yy_rakesh_psf'].values
    xx_gauss=filt['xx_gauss']
    yy_gauss=filt['yy_gauss']

    ax=plt.gca()
    #plt.contourf(snrplot[:,:].T,cmap='pink',levels=[3,5,7,10,14,15])
    plt.pcolormesh(snrplot[:].T,cmap='rainbow')
    plt.colorbar()
    vloc_plot=([25.0+xx_v,25.0+yy_v])
    #c=create_circle(vloc_plot,r=2.69,color='darkcyan',fill=False,label='Valentin ({0:1.2f},{1:1.2f})'.format(np.float(xx_v)*12.5/1000,np.float(yy_v)*12.5/1000) )
    #ax.add_patch(c)
    plt.scatter(vloc_plot[0],vloc_plot[1],marker='*',c='k',label='Valentin')
    #plt.text(vloc_plot[0],vloc_plot[1],"(-{0},-{1})".format(np.round((xx_v*12.5/1000),2),np.round((yy_v*12.5/1000),2)),ha='center',va='center')
    loc_derived=[xx+25,yy+25]
    #c=create_circle(loc_derived,r=2.69,color='indianred',fill=False,label='Rakesh ({0:1.2f},{1:1.2f})'.format(np.float(xx*12.5/1000),np.float(yy*12.5/1000)))
    #ax.add_patch(c)
    plt.scatter(loc_derived[0],loc_derived[1],marker='+',c='k',label='Weighted avg')

    loc_gauss=[xx_gauss+25,yy_gauss+25]

    #c=create_circle(loc_gauss,r=2.69,color='dimgrey',fill=False,label='Gaussian fit ({0:1.2f},{1:1.2f})'.format(np.float(xx_gauss*12.5/1000),np.float(yy_gauss*12.5/1000)))
    #ax.add_patch(c)
    plt.scatter(loc_gauss[0],loc_gauss[1],marker='2',c='k',label='Gaussian avg')
    #plt.text(xx+25,yy+25,"({0},{1})".format(np.round((xx*12.5/1000),2),np.round((yy*12.5/1000),2)),ha='center',va='center')
    #plt.errorbar(xx+25,yy+25,xerr=xx_std,yerr=y_std,c='indianred')
    plt.legend()
    plt.xticks(ax.get_xticks()[0:-1],np.round((np.arange(10,61,10)-35)*12.5/1000,2))
    plt.yticks(ax.get_yticks()[0:-1],np.round((np.arange(10,61,10)-35)*12.5/1000,2))
    plt.title("wmin$={1}$ $\mu$m wmax$={2}$ $\mu$m PCs$={0}$".format(PCs,wmin,wmax),fontweight='bold')
    plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_16.11.2020/SNR_psf_plots_withsymbols/snr_pcs_{0}_wmin_{1}_wmax_{2}.png".format(PCs,wmin,wmax),dpi=300)
    plt.close()