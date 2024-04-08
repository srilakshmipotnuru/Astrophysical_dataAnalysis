import pandas as pd
import numpy as np
from scipy import constants
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split
import emcee
import corner

#read data
df=pd.read_csv('data/Hubble data.csv')
data=np.array([df.z,df.Hz,df.sigma])
z,Hz,sigma=data

#X and y
X=z.reshape(-1,1)
X.shape
y=Hz

#scatter plot
plt.figure(1)
plt.scatter(z,Hz)
plt.savefig("results/Hz_z_scatter.pdf")

#GPR
X_train,X_test,y_train,y_test=train_test_split(X,y)
kernel= gp.kernels.RBF(10.0,(1e-3,1e3))
model=gp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1,normalize_y=True)
model.fit(X_train,y_train)
params=model.kernel_.get_params()

#predicting y
y.shape
y_pred,std=model.predict(X,return_std=True)
model.score(X,y)

#plotting GP with confidence interval
upper,lower=y_pred+1.68 * std, y_pred-1.68*std
plt.errorbar(X,y,fmt='o',label='samples',markersize=5)
plt.plot(X,y_pred,label='Gaussian process',ls='-')
plt.fill_between(X.ravel(),upper,lower,alpha=0.2,label=r"confidence interval",color='#2698eb')
plt.legend()
plt.xlabel("z",fontweight='bold',fontsize=12)
plt.ylabel("Hz",fontweight='bold',fontsize=12)
plt.savefig("results/Hz_vs_z.pdf")


#dc(z) using simpson 3/8 rule
def func(x):
    c=constants.c*(10**-3)
    hz=model.predict(np.array([x]).reshape(-1,1),return_std=False)
    
    return c/hz
def simpson (upper_lim,interval,lower_lim=0):
    interval_size = (float(upper_lim - lower_lim) / interval)
    sum = func(lower_lim) + func(upper_lim)
    for i in range(1, interval ):
        if (i % 3 == 0):
            sum = sum + 2 * func(lower_lim + i * interval_size)
        else:
            sum = sum + 3 * func(lower_lim + i * interval_size)
    #delta= -(1/(80*interval))*((upper_lim-lower_lim)**5)*("#4th derivative") 
    return ((float( 3 * interval_size) / 8 ) * sum )
# calculating Dc    
def Dc(z):
    val=simpson(z,3)
    val2=simpson(z,6) #error using richardsons extrapolation
    
    error=(val2-val)/15
    return [val,error]
    

#calculating dl(z)
def Dl(z):
    dc,err=Dc(z)
    val=(1+z)*dc
    sDl= (1+z)*err
    return[val,sDl]
#dl values for x
dl_val,sig_dl=np.array([Dl(z)[0] for z in X]),np.array([Dl(z)[1] for z in X])

#dl plot
plt.figure(2)
plt.scatter(X,dl_val,s=10,label="samples")
plt.plot(X,dl_val,label='dl values')
plt.title("Dl values")
plt.legend()
plt.xlabel("z",fontweight='bold',fontsize=12)
plt.ylabel("Dl",fontweight='bold',fontsize=12)

plt.savefig("results/Dl values.pdf")

#A220 dataset
df_a220 = pd.read_csv('data/A220GRB.txt',sep = ' ',header=None)
df_a220.columns = ['GRB','z','E_p','S_ep','Sbolo','S_sbolo']
df_a220.sort_values(by = ['z'],inplace=True)

#first 118 values 0<z<2
df_118=df_a220.head(118)
z_val=df_118['z'].to_list()
dl,s_dl=[],[]
for z in z_val:
    temp=Dl(z)
    dl.append(temp[0])
    s_dl.append(temp[1])
df_118['dl'],df_118['s_dl']=dl,s_dl
df_118['Eiso']= np.pi*4*((df_118['dl'])**2)*df_118['Sbolo']/(1+df_118['z'])
df_118['S_Eiso']=df_118['Eiso']*((2*df_118['s_dl']/df_118['dl'])+(df_118['S_sbolo']/df_118['Sbolo']))

#regression
final=pd.DataFrame()
final['Eiso']=np.log10(df_118.Eiso.astype('float'))
final['Eiso_fin']=final['Eiso']+52
final['S_Eiso']=1*df_118.S_Eiso/(2.303*df_118.Eiso)
final['E_p']=np.log10(df_118.E_p.values)
final['S_ep']=1*df_118.S_ep/(2.303*df_118.E_p)

x=final.Eiso_fin.astype('float')
y=final.E_p.astype('float')
yerr=final.S_ep.astype('float')
xerr=final.S_Eiso.astype('float')

a_true = 0.31
b_true = -20.9
s_true = -0.5

def log_likelihood(intparams,x,y,xerr,yerr):
    a,b,err=intparams
    model=a*x+b
    sigma2=err**2+((a**2)*(xerr**2))+ yerr**2
    return -0.5*((np.sum((y-model)**2/sigma2))+np.sum(np.log(2*np.pi*sigma2)))
    
initparams=[a_true,b_true,s_true]
nll=lambda *args: -log_likelihood(*args)
lik_model_1=minimize(nll,initparams,args=(x,y,xerr,yerr))

plt.figure(3)
plt.errorbar(x,y,xerr=xerr,yerr=yerr,ls='',marker='.',color='gray',label='Data')
plt.plot(x,lik_model_1['x'][1] + lik_model_1['x'][0]*x,label="maximum likelihood")
plt.legend()
plt.savefig('results/maxliklihoood-data.pdf')

def log_prior(theta):
    a, b, s = theta
    if -1 <= a <= 1 and -30 <= b <= -10 and 0 <= s < 2:
     return 0.0

    return -np.inf

def log_probability(theta, x, y, xerr, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, xerr, yerr)

#setup for emcee
Nens = 100   # number of ensemble points


m_ini = np.random.uniform(-1,1 , Nens) # initial a points


b_ini = np.random.uniform(-30, -10, Nens) # initial c points

s_ini = np.random.uniform(0,2,Nens)

inisamples = np.array([m_ini, b_ini,s_ini]).T # initial samples

ndims = inisamples.shape[1] # number of parameters/dimensions
Nburnin = 500   # number of burn-in samples
Nsamples = 750 #final posterior samples


#print('emcee version: {}'.format(emcee.__version__))

# for bookkeeping set number of likelihood calls to zero
log_likelihood.ncalls = 0

# set additional args for the posterior (the data, the noise std. dev., and the abscissa)
argslist = (x,y,xerr,yerr)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, log_probability, args=argslist)    

from time import time
t0 = time() # start time
sampler.run_mcmc(inisamples, 2000,progress=True)
t1 = time()

timeemcee = (t1-t0)
print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

# extract the samples (removing the burn-in)
samples_emcee = sampler.get_chain(flat=True, discard=Nburnin)   

CORNER_KWARGS = dict(
    smooth = 0.9,
    label_kwargs=dict(fontsize=16,fontweight = 'bold'),
    title_kwargs=dict(fontsize=16),
    levels=(0.68,0.90,0.95),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=True,
    max_n_ticks=5,fontweight='bold',fontsize = 12
    # ,title_fmt=".2E"
    ,label_fmt='.2f'
)
def plotposts(samples, **kwargs):
    fig = corner.corner(samples, labels=['m','c','sig s'],**CORNER_KWARGS,  **kwargs)
    
plotposts(samples_emcee)
plt.savefig("results/amati_parameters.pdf")
    
    
