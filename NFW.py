import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import emcee

#prima richiesta

#definizione della costante densità

rho_m=0.286*(0.7)**2*2.7751428946e11

#definizione delle functions per calcolare il profilo di densità sigma

def r_s(M,c,z):
    y=((M*3/(4*np.pi*200*rho_m*(1+z)**3))**(1.0/3))*(1/c)
    return y

def rho_s(c,z):
    fc=np.log(1+c)-(c/(1+c))
    delta_char=(200*(c**3))/(3*fc)
    y=rho_m*((1 + z)**3)*delta_char
    return y

def f(R,M,c,z):
    
    x=R/r_s(M,c,z) #la lunghezza di x dipenderà dal partizionamento (arbitrario) dell'intervallo in cui R è definito
    
    g=np.zeros(len(x)) #definizione di un array g di dimensione la lunghezza di x
    
    for j in range(len(x)): #assegnazione dei valori per i diversi casi
        if x[j]<1:
            g[j]=(1.0-(2.0*np.arctanh(np.sqrt((1-x[j])/(1+x[j])))/np.sqrt(1-x[j]**2)))*(1/(x[j]**2-1))
        elif x[j]==1:
            g[j]= 1.0/3
        elif x[j]>1:
            g[j]=(1.0-(2.0*np.arctan(np.sqrt((x[j]-1)/(1+x[j])))/np.sqrt(x[j]**2-1)))*(1/(x[j]**2-1))
    return g

#definizione finale del profilo di densità

def sigma(R,M,c,z):
    y=2.0*r_s(M,c,z)*rho_s(c,z)*f(R,M,c,z)
    return y

#partizionamento di R per definire x
R=np.linspace(0.03, 1.0, 150)  #suddivido l'intervallo di R in 150 valori

#assegnazione dei valori ai parametri liberi

M=[10**13.5, 10**14.0, 10**15.0]
c=[2, 5, 10]
z=0.0


#utilizzo subplots per avere tutti e 9 i grafici con una sola call

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14,9), sharex=True)
for i in range(3):
    for j in range(3):
        ax[i,j].set_xscale("log") #metto i grafici in scala log-log
        ax[i,j].set_yscale("log")
        
        y=sigma(R,M[i],c[j],z)
        ax[i,j].plot(R,y,color='Red') #plotto sulle x i valori di R e sulle y la sigma
        ax[i,j].annotate(f"$c$={c[j]}  $logM$={np.log10(M[i])}", xy=(R[0], y[-2]), fontsize=12)
        #con xy=... specifico la posizione del testo dell'annotate, riportando i valori di c e m

for i in range(3):
    ax[i, 0].set_ylabel('Profilo Σ [M$_\odot$ / Mpc$^2$]')
    ax[2, i].set_xlabel("R [Mpc]")
    
plt.show()


#seconda richiesta

#apro i file
data_set=fits.getdata('halo_catalog.fit') 
#print(np.shape(data_set)) 
data_set_R=np.load("R_values.npy")
#print(np.shape(data_set_R))

#seleziono le colonne
redshift=data_set['Z']
ricchezza=data_set['LAMBDA_TR']
prof_den=data_set['SIGMA_of_R']

#definisco in una lista gli estremi dell'intervallo
#della ricchezza
estremi=[15,20,30,45,60,200]

#per ogni intervallo calcolo il redshift, il profilo
#di densità medio, la dev. std. e grafico il profilo 
#in funzione di R, con relativa dev. std.


#definisco un array di 5 elementi dove allocherò
#i valori medi di redshift
redshift_medio=np.zeros(5)
prof_den_medio=np.zeros((5,8)) #una matrice per i profili medi
dev_std_prof=np.zeros((5,8)) #una matrice per le relative dev. std.


#analogamente a prima, per fare una sola call dei 5 grafici, uso subplots
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8,13), layout='constrained',sharex=True)


for j in range(5):
    data=data_set[ (ricchezza<=estremi[j+1]) & (ricchezza>=estremi[j])]
    #seleziono le varie ricchezze nei 5 intervalli
    redshift_medio[j]=np.mean(data['Z']) #calcolo la media dei redshift
    prof_den_medio[j]=np.mean(data['SIGMA_of_R'],axis=0) #media dei profili
    dev_std_prof[j]=np.std(data['SIGMA_of_R'],axis=0) #dev. std. dei profili
    #axis=0 mi specifica di fare l'operazione lungo la prima riga della matrice
    #che nel ciclo for corrisponde ai valori nel j-esimo intervallo relativo
    
    #anche in questo caso mi pongo in una scala log-log per meglio apprezzare i dati
    ax[j].set_xscale("log")
    ax[j].set_yscale("log")
    #metto la dev. std. sui valori medi dei profili
    ax[j].errorbar(data_set_R, prof_den_medio[j], yerr=dev_std_prof[j], fmt='o',color='black')
    ax[j].set_title(f'Intervallo n°{j+1} dei dati')
    ax[j].set_ylabel('Profilo Σ medio [M$_\odot$ / Mpc$^2$]')
    ax[j].set_ylim([10**13,10**16]) #ho messo un limite di visualizzazione sull'asse y
ax[-1].set_xlabel('R [Mpc]')

plt.show()
#stampo a schermo anche i valori medi dei redshift
print("I valori medi dei redshift sono:",redshift_medio)
   


#terza richiesta


mat_cov=np.load('covariance_matrices.npy')

def ln_Prior(theta): #definizione del prior uniforme
    logM,c=theta
    if (0.2<=c<=20 and 13.0<=logM<=16.0):
        return 0.0 
    return -np.inf

def ln_Likelihood(theta, R, dati, m_cov, z): #def. la likelihood gaussiana
    logM, c = theta
    M = 10**(logM) #per usare poi il 'modello', a cui passo M e non logM
    modello = sigma(R, M, c, z)
    #necessito della matrice inversa della covarianze per definire la likelihood gaussiana
    mat_cov_inversa = np.linalg.inv(m_cov)
    y=-(1.0/2)*np.dot(dati-modello,np.dot(mat_cov_inversa, dati-modello))
    return y 
#ci dovrebbe essere una costante spezzata in un termine additivo dal logaritmo
    
def ln_Posterior(theta, R, dati, m_cov, z): 
    logM,c=theta
    lp=ln_Prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_Likelihood(theta, R, dati, m_cov, z)
#il logaritmo spezza in somma il prodotto che definisce la posterior


from scipy.optimize import minimize
best_fit = np.zeros((5,2)) #poiché sono 5 intervalli di ricchezza
np.random.seed(42)
data_chain = np.zeros((5,28800,2)) #per poi salvare la chain

for j in range(5):
    nll = lambda *args: -ln_Posterior(*args)
    initial = np.array([15.0, 3.0])*( 1 + 0.1*np.random.randn(2))
    soln=minimize(nll, initial, args=(data_set_R, prof_den_medio[j], mat_cov[j], redshift_medio[j]))
    logM,c=soln.x
    best_fit[j]=soln.x
    
    pos = soln.x *(1 + 1e-4 * np.random.randn(32,2))
    nwalker, ndim = pos.shape
    
    sampler = emcee.EnsembleSampler(nwalker, ndim, ln_Posterior, args=(data_set_R, prof_den_medio[j], mat_cov[j], redshift_medio[j]))
    sampler.run_mcmc(pos, 1000, progress=True)
    #ho dovuto ridurre il numero di step per crash improvvisi del computer nell'esecuzione
    data_chain[j] = sampler.get_chain(discard=100 , flat=True) #alloco e salvo i dati
    
    fig, ax = plt.subplots(2, figsize=(11,5), layout='constrained', sharex=True)
    samples = sampler.get_chain()
    labels=["$logM$","$c$" ]
    for k in range(ndim):
        AX = ax[k]
        AX.plot(samples[:,:,k], "k", alpha=0.3)
        AX.set_xlim(0,len(samples))
        AX.set_ylabel(labels[k])
        AX.yaxis.set_label_coords(-0.1, 0.5)
    ax[-1].set_xlabel("step numbers");

#quarta richiesta

#mostro i valori di best fit e plotto il grafico triangolare
import pygtc
for j in range(5):
    print(f"Valori di logM e c per l'intervallo n°{j+1}:", best_fit[j]) 
    GTC = pygtc.plotGTC(chains=data_chain[j], paramNames=['$logM$', '$c$'],
                       chainLabels=[f'intervallo n°{j+1}'], figureSize='MNRAS_page')

#calcolo la dev. std. e la media dei dati presi dalle catene precedenti
#per poi confrontarli
c_medio_cat=np.zeros(5)
c_dev_std_cat=np.zeros(5)
logM_medio_cat=np.zeros(5)
logM_dev_std_cat=np.zeros(5)


for j in range(5):
    c_medio_cat[j] = np.mean(data_chain[j,:,1])
    c_dev_std_cat[j] = np.std(data_chain[j,:,1])
    logM_medio_cat[j] = np.mean(data_chain[j,:,0])
    logM_dev_std_cat[j] = np.std(data_chain[j,:,0])
    print(f'intervallo n°{i+1}')
    print("Valor medio e dev. std. di logM:", logM_medio_cat[j],logM_dev_std_cat[j])
    print("Valor medio e dev. std. di c:", c_medio_cat[j], c_dev_std_cat[j])
    print()

#Confrontiamo i dati ottenuti dal best fit e quelli delle catene di cui
#riportiamo anche gli errori, per tutti i 5 intervalli di ricchezza

fig,ax = plt.subplots(2, figsize=(8,8), layout='constrained')
for j in range(5):
    ax[0].plot(j+1, best_fit[j,0], 'o', color='b')
    ax[0].errorbar(j+1, logM_medio_cat[j], logM_dev_std_cat[j], xerr=None, color='r', fmt='o')
    ax[1].plot(j+1, best_fit[j,1], 'o', color='b')
    ax[1].errorbar(j+1, c_medio_cat[j], c_dev_std_cat[j], xerr=None, color='r', fmt='o')

ax[0].set_ylabel("$logM$")
ax[1].set_ylabel("$c$")
ax[-1].set_xlabel("Intervallo di ricchezza n°")
plt.show()

#Calcolo del chi quadro per i 5 intervalli

chi_quadro=np.zeros(5)

for i in range(5):
    chi_quadro[i] = sum((prof_den_medio[i] - sigma(data_set_R, 10**best_fit[i,0], best_fit[i,1], redshift_medio[i]))**2/((dev_std_prof[i])**2))
    

print("I valori ottenuti di chi quadro sono:", chi_quadro)


#Confronto dei profili di densità ottenuti dal best fit con l'andamento dei profili simulati forniti dal file 

#procedo sempre con subplots per fare tutto con una call

fig, ax = plt.subplots(5, 1, figsize=(8,16), layout='constrained')
for i in range(5):
    ax[i].set_yscale("log")
    ax[i].set_xscale("log")
    ax[i].plot(R, sigma(R, 10**best_fit[i,0], best_fit[i,1], redshift_medio[i]), color='red')
    ax[i].errorbar(data_set_R, prof_den_medio[i], dev_std_prof[i], fmt='o', color='b')
    ax[i].set_ylabel('Profilo Σ medio [M$_\odot$ / Mpc$^2$]')
    ax[i].set_title(f'Intervallo n°{i+1}: best fit e andamento dei dati iniziali')
    ax[i].set_ylim([10**13,10**15])
ax[-1].set_xlabel('R [Mpc]')
plt.show()

#In ultima analisi il confronto dei dati di best fit con 100 valori 
#di M e c estratti dalle catene di cui sopra


fig, ax = plt.subplots(5, 1, figsize=(9,15), layout='constrained')
indici = list(range(0,28800)) 

for i in range(5): #alloco per estrarre poi 100 valori casuali dalle catene
    logM_random = np.zeros(100)
    c_random = np.zeros(100)
    profilo_cat = np.zeros((8,100)) #mi serve dopo per allocare i profili calcolati con i valori estratti
    for j in range(100):
        #pesco senza ripeterli 100 numeri casuali dall'array indici
        indici_random = np.random.choice(indici, size=100, replace=False) 
        logM_random[j] = data_chain[i,indici_random[j],0]
        c_random[j] = data_chain[i,indici_random[j],1]
        #calcolo i valori estratti dalle catene con il profilo di densità sigma e li alloco
        profilo_cat[:,j] = sigma(data_set_R, 10**(logM_random[j]), c_random[j], redshift_medio[i]) 
    
    #passo a graficare i rispettivi 100 profili ottenuti sopra, sempre in scala log-log
    
    
    ax[i].set_yscale("log")
    ax[i].set_xscale("log")
    ax[i].plot(data_set_R, profilo_cat, color='c', alpha=0.10) #quelli in azzurro chiaro sono i 100 profili
    ax[i].errorbar(data_set_R, prof_den_medio[i], dev_std_prof[i], fmt='o', color='m') #plotto i valori medi e relativi errori
    ax[i].plot(R, sigma(R, 10**best_fit[i,0], best_fit[i,1], redshift_medio[i]), color='red', lw=2) #plotto il best fit
    ax[i].set_title(f'Intervallo n°{i+1}: confronto dei dati')
    ax[i].set_ylabel('Profilo Σ medio [M$_\odot$ / Mpc$^2$]')
    ax[i].set_ylim([10**13,10**15.5])
ax[-1].set_xlabel('R [Mpc]')
plt.show()