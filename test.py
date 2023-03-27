
import itertools
from PIL import Image
from numpy import asarray
from os import path
import numpy as np
import soundfile as sd
import itertools
import wave

from scipy.io.wavfile import read
from scipy.io.wavfile import write

#a = read(r"C:\Users\Achra\Downloads\Thunderclouds [Acoustic Version] - LSD ft. Sia, Diplo, Labrinth.wav",mmap=True)
#samplerate = 2*a[0]
#a=np.array(a[1])
#print(a)




#data=list(itertools.chain.from_iterable(a))
#data=np.array(data)



#writing file

#m=np.max(data)

#data32 = (data/m).astype(np.float64)

#write('test50.wav',samplerate, data32 )

L=np.array([1,2,3,6,47,9])
M=np.array([1,7,2])
L=np.concatenate((L,M))
print(L)



#image1 = Image.open(r"C:\Users\Achra\Desktop\Hraf\S7\Projet scientique\Projet_Scientifique_BSS\Projet_Scientifique_BSS\im1.png")

#image1.show()