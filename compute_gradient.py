import numpy as np

#Cette fonction calcul le gradient du critère pénalisé : 
#information mutuelle + lambda pénalisation
#où lambda est un hyperparamètre de pénalisation
#l'information mutuelle caractérise l'indépendance
#la pénalisation normalise le vecteur et le force à avoir 
#un écart-type = 1 


def compute_grad(y1,y2,x1,x2):
    #Estimation polynomiale de la fonction score marginale (Psi) par moindres carrés
    #Thèse Massoud Babaie Zadeh page 46       
    m_y1=np.mean(y1)
    m_y12=np.mean(y1**2)
    m_y13=np.mean(y1**3)
    m_y14=np.mean(y1**4)
    m_y15=np.mean(y1**5)
    m_y16=np.mean(y1**6)
    m_y2=np.mean(y2)
    m_y22=np.mean(y2**2)
    m_y23=np.mean(y2**3)
    m_y24=np.mean(y2**4)
    m_y25=np.mean(y2**5)
    m_y26=np.mean(y2**6)

    #Calcul des différents parametres (approximation des fonctions scores par moindres carrées)

    M1=np.array([[1 ,m_y1 ,m_y12 ,m_y13],[m_y1 ,m_y12 ,m_y13 ,m_y14],[m_y12 ,m_y13 ,m_y14 ,m_y15],[m_y13 ,m_y14 ,m_y15 ,m_y16]])
    M2=np.array([[1, m_y2 ,m_y22 ,m_y23],[m_y2 ,m_y22 ,m_y23 ,m_y24],[m_y22 ,m_y23 ,m_y24 ,m_y25],[m_y23 ,m_y24 ,m_y25 ,m_y26]])
    P1=np.array([0 ,1 ,2*m_y1 ,3*m_y12])
    P2=np.array([0 ,1 ,2*m_y2 ,3*m_y22])
    w1=np.dot(-np.linalg.inv(M1),P1)
    w2=np.dot(-np.linalg.inv(M2),P2)

    #calcul du Psi
    Psi_y1=w1[0]+w1[1]*y1+w1[2]*(y1**2)+w1[3]*(y1**3)
    Psi_y2=w2[0]+w2[1]*y2+w2[2]*(y2**2)+w2[3]*(y2**3)
    

    #Calcul de la Jacobienne de l'information mutuelle
    M_Psi11=np.mean(Psi_y1*x1)
    M_Psi12=np.mean(Psi_y1*x2)
    M_Psi21=np.mean(Psi_y2*x1)
    M_Psi22=np.mean(Psi_y2*x2)
    sep=np.array([[M_Psi11 ,M_Psi12],[M_Psi21 ,M_Psi22]])
    #J'enlève la moyenne
    y1=y1-np.mean(y1)
    y2=y2-np.mean(y2)
    #Calcul de la Jacobienne de la normalisation (penalisation) 
    #papier Mohammed El Rhabi 2004
    temp1=4*(np.mean(y1**2)-1)*y1
    temp2=4*(np.mean(y2**2)-1)*y2
    m_y1x1=np.mean(temp1*x1)
    m_y1x2=np.mean(temp1*x2)
    m_y2x1=np.mean(temp2*x1)
    m_y2x2=np.mean(temp2*x2)
    pen=np.array([[m_y1x1 ,m_y1x2],[m_y2x1 ,m_y2x2]])
    L=[sep,pen]

    return L
