from math import log10
from statistics import fmean
from decompose import decomposer
from compute_gradient import compute_grad
from PIL import Image
import numpy as np
from numpy import asarray
from correl_coef_composante_nb import correl_coef_composante
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write


#image1 = Image.open(r"C:\Users\Achra\Desktop\Hraf\S7\Projet scientique\Projet_Scientifique_BSS\Projet_Scientifique_BSS\im1.png")
#image2 = Image.open(r"C:\Users\Achra\Desktop\Hraf\S7\Projet scientique\Projet_Scientifique_BSS\Projet_Scientifique_BSS\im2.png")
#img1 = asarray(image1)
#img2 = asarray(image2)

#image_vecteur_s1=decomposer(img1)[0]
#nbligne_s1 = decomposer(img1)[1]
#nbcolonne_s1 = decomposer(img1)[2]
#image1.show()

#image_vecteur_s2=decomposer(img2)[0]
#nbligne_s2 = decomposer(img2)[1]
#nbcolonne_s2 = decomposer(img2)[2]
#image2.show()



#Création de la matrice de melange X = A* S,A est l'opérateur de mélange
#  dans la "vraie vie", il est inconnu, ici pour tester l'algorithme on le simule

A=np.array([[0.6 ,0.4],[0.4 ,0.6]])


#s1=image_vecteur_s1
#s2=image_vecteur_s2


a_s1 = read(r"C:\Users\Achra\Desktop\Hraf\BSS_python\Thunderclouds original [music].wav")
samplerate1 = 2*a_s1[0]
a_s1=np.array(a_s1[1])



a_s2 = read(r"C:\Users\Achra\Desktop\Hraf\BSS_python\Thunderclouds original [vocals].wav")
samplerate2 = 2*a_s2[0]
a_s2=np.array(a_s2[1])

audio_s1=decomposer(a_s1)[0]
nbligne_s1 = decomposer(a_s1)[1]
nbcolonne_s1 = decomposer(a_s1)[2]

audio_s2=decomposer(a_s2)[0]
nbligne_s2 = decomposer(a_s2)[1]
nbcolonne_s2 = decomposer(a_s2)[2]

s1=audio_s1
s2=audio_s2
#l1=len(s1)
#l2=len(s2)
#s1=np.concatenate((s1,s2[l1:l2+1]))



#n normalise
x1=s1/np.std(s1)
x2=s2/np.std(s2)


s1=(x1/np.max(x1)).astype(np.float64)

plt.plot(np.arange(1,len(s1)+1), s1)
plt.xlabel("t")
plt.ylabel("s1")
plt.legend()
plt.show()

s2=(x2/np.max(x2)).astype(np.float64)

plt.plot(np.arange(1,len(s2)+1), s2)
plt.xlabel("t")
plt.ylabel("s2")
plt.legend()
plt.show()



m1=A[0,0]*s1+A[0,1]*s2

plt.plot(np.arange(1,len(m1)+1), m1)
plt.xlabel("t")
plt.ylabel("m1")
plt.legend()
plt.show()

m2=A[1,0]*s1+A[1,1]*s2


plt.plot(np.arange(1,len(m2)+1), m2)
plt.xlabel("t")
plt.ylabel("m2")
plt.legend()
plt.show()


# vecteur --> matrice
#calcul préliminaire
#xx1=(x1*np.std(x1))
#xx2=(x2*np.std(x2))
#X1=max(abs(xx1))
#X2=max(abs(xx2))
#xx1=xx1/X1*255
#xx2=xx2/X2*255


#print(xx1)
#reconstruction de nos siganux melangées
#image_matrice_s1=Image.fromarray(xx1.reshape(nbligne_s1,nbcolonne_s1))
#image_matrice_s1.show()

#image_matrice_s2=Image.fromarray(xx2.reshape(nbligne_s2,nbcolonne_s2))

#image_matrice_s2.show()

#normalisation des deux sorties mélangées



write('melange1.wav',samplerate2, m1 )


write('melange2.wav',samplerate2, m2 )



xx1=m1/np.max(m1)
xx2=m2/np.max(m2)

#création de la matrice de séparation
nb_iter=200
B=np.eye(2)
mu=0.05   
lam=0.001
y1=m1
y2=m2
RSBB1=np.array([])
RSBB2=np.array([])
for i in range(nb_iter+1):
    print(i)
    GradIM = compute_grad(y1,y2,m1,m2)[0]
    Gradpen = compute_grad(y1,y2,m1,m2)[1]
    B=B-mu*(-np.dot(B,GradIM)-np.eye(2))-mu*lam*Gradpen
    

    #calcul de la separation
    y1=B[0,0]*m1+B[0,1]*m2
    y2=B[1,0]*m1+B[1,1]*m2
    
    #calcul fictif pour recuperer la moyenne et la variance
    yy1=B[0,0]*xx1+B[0,1]*xx2
    yy2=B[1,0]*xx1+B[1,1]*xx2
    #m_yy1=mean(yy1)
    #m_yy2=mean(yy2)
    e_yy1=np.std(yy1)
    e_yy2=np.std(yy2)
    BA=np.dot(B,A)
    bruit1=BA[0,1]*s2
    bruit2=BA[1,0]*s1

   
    y1est = y1;             #pour vérifier les écarts types
    y2est = y2;             #rappel : on cherche les sources 
    
    RSBB1 = np.append(RSBB1,[10*log10(np.mean(y1est**2)/np.mean(bruit1**2))])                      #ayant un écart type =1
    RSBB2 = np.append(RSBB2,[10*log10(np.mean(y2est**2)/np.mean(bruit2**2))])


plt.plot(np.arange(1,len(y1est)+1), y1est )
plt.xlabel("t")
plt.ylabel("y1est ")
plt.legend()
plt.show()


plt.plot(np.arange(1,len(y2est)+1), y2est )
plt.xlabel("t")
plt.ylabel("y1est ")
plt.legend()
plt.show()







t = np.arange(1,nb_iter+2)

plt.plot(t, RSBB1, color='r', label='SNR 1')
plt.plot(t, RSBB2, color='g', label='SNR 2')

plt.xlabel("Nombre d'iterations")
plt.ylabel("SNR")
plt.title("Courbe SNR(i)")

plt.legend()
plt.show()
#Calcul des matrice de correlation 
#rappel : lorsque deux signaux sont indépendants ou décorrélés
#Cette matrice est la matrice identité

#[Mat_cor] = correl_coef_composante(s1,s2)
#[Mat_cor] = correl_coef_composante(x1,x2)
#[Mat_cor] = correl_coef_composante(y1est,y2est)

#reconstruction des matrices pour obtenir les images séparées
#cette reconstruction est obtenu avec l'aide de la variance et 
# la moyenne des images de mélanges

y1=y1*e_yy1
y2=y2*e_yy2

#V1=(y1/max(y1))*255
#V2=(y2/max(y2))*255

V1=(y1/np.max(y1)).astype(np.float64)
write('separation1.wav',samplerate2, V1 )

V2=(y2/np.max(y2)).astype(np.float64)
write('separation2.wav',samplerate2, V2 )

#imagematrices1=Image.fromarray(V1.reshape(nbligne_s1,nbcolonne_s1))
#imagematrices1.show()

#imagematrices2=Image.fromarray(V2.reshape(nbligne_s2,nbcolonne_s2))
#imagematrices2.show()






#calcul des rapports signal/bruit(diaphonie)
#BA=np.dot(B,A)
#bruit1=BA[0,1]*s2
#bruit2=BA[1,0]*s1
#RSBB1=10*log10(np.mean(y1est*2)/np.mean(bruit1*2))
#RSBB2=10*log10(np.mean(y2est*2)/np.mean(bruit2*2))