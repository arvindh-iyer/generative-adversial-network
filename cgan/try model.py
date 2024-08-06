from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

g_model=load_model('generator.h5')

def test(number):
    r=c=5
    noise=np.random.normal(loc=0,scale=1,size=(r*c,100))
    cl=np.ones(shape=(25,))*number
    gen_images= g_model.predict([noise,cl])
    gen_images=(gen_images+1.)/2.
    # print(gen_images)
    fig,ax=plt.subplots(5,5)
    cnt=0
    for i in range(r):
        for j in range(c):
            ax[i,j].imshow(gen_images[cnt,:,:,0],cmap='gray',vmin=0,vmax=1)
            ax[i,j].axis('off')
            cnt+=1
    fig.savefig(f"cimages3/{number}")
    plt.close()

for i in range(0,10):
    test(i)
