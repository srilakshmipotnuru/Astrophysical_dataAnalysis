import pandas as pd
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split
df=pd.read_csv('Hubble data.csv')
data=np.array([df.z,df.Hz,df.sigma])
z,Hz,sigma=data

x=z.reshape(-1,1)
x.shape
y=Hz
plt.figure(1)
plt.scatter(z,Hz)
plt.savefig("results/Hz_z_scatter.pdf")
x_train,x_test,y_train,y_test=train_test_split(x,y)
kernel= gp.kernels.ConstantKernel(1.0,(1e-1,1e3))*gp.kernels.RBF(10.0,(1e-3,1e3))
model=gp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1,normalize_y=True)
model.fit(x_train,y_train)
params=model.kernel_.get_params()

y.shape
y_pred,std=model.predict(x,return_std=True)
model.score(x,y)

upper,lower=y_pred+1.68 * std, y_pred-1.68*std
plt.errorbar(x,y,fmt='o',label='samples',markersize=5)
plt.plot(x,y_pred,label='Gaussian process',ls='-')
plt.fill_between(x.ravel(),upper,lower,alpha=0.2,label=r"confidence interval",color='#2698eb')
plt.legend()
plt.xlabel("z",fontweight='bold',fontsize=12)
plt.ylabel("Hz",fontweight='bold',fontsize=12)
plt.savefig("results/Hz_vs_z.pdf")


#dc(z) using simpson 3/8 rule
def func(x):
    c=constants.c*(10**-3)
    hz,sigmaHz=model.predict(np.array([x]).reshape(-1,1),return_std=True)
    print(sigmaHz)
    return c/hz
def Dc(upper_lim,lower_lim=0,interval=100):
    
    interval_size = (float(upper_lim - lower_lim) / interval)
    sum = func(lower_lim) + func(upper_lim)
    for i in range(1, interval ):
        if (i % 3 == 0):
            sum = sum + 2 * func(lower_lim + i * interval_size)
        else:
            sum = sum + 3 * func(lower_lim + i * interval_size)
     
    return ((float( 3 * interval_size) / 8 ) * sum )

#calculating dl(z)
def Dl(z):
    dc=Dc(z)
    return((1+z)*dc)
#dl values for x
dl_val=np.array([Dl(z) for z in x])
plt.figure(2)
plt.scatter(x,dl_val,s=10,label="samples")
plt.plot(x,dl_val,label='dl values')
plt.title("Dl values")
plt.legend()
plt.xlabel("z",fontweight='bold',fontsize=12)
plt.ylabel("Dl",fontweight='bold',fontsize=12)
plt.savefig("results/Dl values.pdf")

#A220 dataset
df_a220 = pd.read_csv('A220GRB.txt',sep = ' ',header=None)
df_a220.columns = ['GRB','z','E_p','S_ep','Sbolo','S_sbolo']
df_a220.sort_values(by = ['z'],inplace=True)
print(df_a220)

    


