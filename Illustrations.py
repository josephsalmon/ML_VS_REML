from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

### Construction du graphique des deux gaussiennes

x_min = -5.0
x_max = 10

mean = 0 
std = 1.0

mean2 = 1.0
std2 = 2.0

x = np.linspace(x_min, x_max, 100)

y1 = scipy.stats.norm.pdf(x,mean,std)
y2 = scipy.stats.norm.pdf(x,mean2,std2)

plt.plot(x,y1, color='coral', label = 'distribution normale centrée réduite')
plt.plot(x,y2, color='blue', label = 'distribution avec des estimateurs biaisés')

plt.grid()

plt.xlim(x_min,x_max)
plt.ylim(0,0.5)

plt.title('Comparaison de deux gaussiennes',fontsize=10)

plt.legend()

plt.xlabel('x')
plt.ylabel('Densité')

plt.savefig("normal_distribution.png")

###  Création du jeu de données proposé dans l'article 

data = np.array([[0, 10, 1], [1, 25, 1], [0, 3, 2], [1, 6, 2]])
df = pd.DataFrame(data, columns = ['Treat', 'Resp', 'Ind'])

### représentation graphique du jeu de données proposé dans l'article 

plt.style.use('ggplot')
plt.scatter(df.Treat, df.Resp, c=df.Ind)
plt.plot(df.Treat.iloc[0:2], df.Resp.iloc[0:2], '-', c='purple', label = '1')
plt.plot(df.Treat[2:4], df.Resp.iloc[2:4], '-', c='yellow', label='2')
plt.xlabel('Treat')
plt.ylabel('Resp')
plt.legend(title = 'Ind')
plt.savefig("points.png")

### modèle linéaire par régression des moindres carrés (OLS)

results_ols = smf.ols('Resp ~ Treat', data=df).fit()
results_ols.summary()

# Le maximum de vraisemblance vaut -13.549
# beta1 = 6.5 
# beta2 = 6.5 + 9 = 15.5


### Modèle Linéaire mixte par ML

model_mixte_ml = smf.mixedlm("Resp ~ Treat", df, groups = df['Ind'])
result_ml = model_mixte_ml.fit(reml=False)
print(result_ml.summary())

# Le maximum de vraisemblance vaut -13.0029
# on trouve les mêmes valeurs de beta
# de plus on note que sigma^2 = sqrt(18) = 4.24
# sigma^2_s = sqrt(33.25) = 5.77


### Modèle linéaire mixte par REML

model_mixte_reml = smf.mixedlm('Resp ~ Treat', data=df, groups = df['Ind'])
result_reml = model_mixte_reml.fit()
result_reml.summary()

# Le maximum de vraisemblance vaut -7.8877
# la différence entre les deux vraisemblances est expliquée dans le rapport
# on trouve les mêmes valeurs de beta
# de plus on note que sigma^2 = sqrt(36) = 6
# sigma^2_s = sqrt(66.5) = 8.15


### Calcul de la log-vraisemblance

def f(x):
   sigma = x[0]
   sigmas = x[1]
   beta1 = 6.5
   beta2 = 15.5
   y11 = 3
   y12 = 10
   y21 = 6
   y22 = 25
   return(-(1/2)*np.log(4*sigmas**4*sigma**4 +
                        4*sigmas**2*sigma**6 + sigma**8) -(1/2)*np.log(4/((sigma**2)*(sigma**2+2*sigmas**2)))- 
          (1/2)*(1/((sigma**2)*(sigma**2+2*sigmas**2)))*(((y11-beta1)**2)*(sigma**2+sigmas**2) - 
                                                         2*(y11-beta1)*(y21-beta2)*(sigmas**2) + 
                                                         ((y21-beta2)**2)*(sigma**2+sigmas**2) + 
                                                         ((y12-beta1)**2)*(sigma**2+sigmas**2) - 
                                                         2*(y12-beta1)*(y22-beta2)*(sigmas**2) + 
                                                         ((y22-beta2)**2)*(sigma**2+sigmas**2)))

### Maximisation de la log-vraisemblance

sigma_chap=0 # on initialise \hat{\sigma^2}
sigma_s_chap=0 # on initialise \hat{\sigma_s^2}
LL = -10.0 # on initialise la log-vraisemblance
liste = np.arange(-10, 10, 0.01) #on prend un pas de 0.01 pour sigma_s^2
#un pas de 1 suffit pour sigma^2
for x in range(-10,10):
    for y in liste:
        if LL < f([x,y]) : 
           sigma_chap = x
           sigma_s_chap = y
           LL = f([x,y])
    
print("Le maximum de la log vraisemblance est",LL)
print("sigma^2 vaut",sigma_chap, "et sigma_s^2 vaut",sigma_s_chap)

# On trouve les bonnes valeurs de sigma 
