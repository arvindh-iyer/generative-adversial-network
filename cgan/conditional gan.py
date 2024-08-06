import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Embedding,Input,Dense,Reshape,Concatenate,Conv2D,LeakyReLU,Flatten,Dropout,Conv2DTranspose
from keras import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt



def build_discriminator(n_classes,img_shape=(28,28,1)):
    in_class=Input(shape=(1,))
    in_vector=Embedding(input_dim=n_classes,output_dim=50)(in_class)
    layer=Dense(units=np.prod(img_shape),activation='tanh')(in_vector)
    class_input=Reshape(target_shape=img_shape)(layer)

    in_image=Input(shape=img_shape)
    input_total=Concatenate()([in_image,class_input])
    layers=Conv2D(128,(3,3),strides=(2,2),padding='same')(input_total)
    layers=LeakyReLU(0.2)(layers)
    layers=Conv2D(128, (3, 3), strides=(2, 2), padding='same')(layers)
    layers=LeakyReLU(0.2)(layers)
    layers=Flatten()(layers)
    layers=Dropout(0.4)(layers)
    layers=Dense(units=1,activation='sigmoid')(layers)

    model=Model([in_image,in_class],layers)
    opt=Adam(0.002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model

def build_generator(n_classes=10):
    in_class = Input(shape=(1,))
    in_vector = Embedding(input_dim=n_classes, output_dim=50)(in_class)
    layer=Dense(8*8*1)(in_vector)
    layer=Reshape(target_shape=(8,8,1))(layer)

    noise=Input(shape=(100,))
    layer2=Dense(8*8*128)(noise)
    layer2=Reshape(target_shape=(8,8,128))(layer2)

    final_input=Concatenate()([layer2,layer])
    layers=Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding='same')(final_input)
    layers=LeakyReLU(0.2)(layers)
    layers = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(layers)
    layers = LeakyReLU(0.2)(layers)
    layers=Conv2D(1,kernel_size=(5,5),padding='valid',activation='tanh')(layers)

    model=Model([noise,in_class],layers)
    return model


d_model=build_discriminator(10,(28,28,1))
g_model=build_generator(10)

def gan_model(d_model,g_model):
    d_model.trainable=False
    g_noise,g_class=g_model.input
    # g_image=g_model.output
    g_image=g_model([g_noise,g_class])
    d_output=d_model([g_image,g_class])
    combined=Model([g_noise,g_class],d_output)
    opt = Adam(lr=0.002, beta_1=0.5)
    combined.compile(loss='binary_crossentropy', optimizer=opt)
    return combined

combined=gan_model(d_model,g_model)

def generate_real_samples(x,y,batch_size):
    index=np.random.randint(0,x.shape[0],batch_size)
    x_data=x[index]
    y_data=y[index]
    return x_data,y_data

def generate_fake_samples(batch_size):
    x_data=np.random.normal(0,1,size=(batch_size,100))
    y_data=np.random.randint(0,9,size=(batch_size,1))
    return x_data,y_data

def save(epoch):
    r=c=5
    noise=np.random.normal(loc=0,scale=1,size=(r*c,100))
    cl=np.ones(shape=(25,))*3
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
    fig.savefig(f"cimages2/img{epoch}")
    plt.close()


def training(epochs,batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=(x_train-127.5)/127.5
    half_size=int(batch_size/2)
    for epoch in range(epochs):
        real_x, real_y = generate_real_samples(x_train, y_train, half_size)
        fake_x, fake_y = generate_fake_samples(half_size)
        fake_img=g_model.predict([fake_x,fake_y])
        d_loss_real=d_model.train_on_batch([real_x,real_y],np.ones(shape=(half_size,)))
        d_loss_fake=d_model.train_on_batch([fake_img,fake_y],np.zeros(shape=(half_size,)))
        # print(d_loss_fake)
        # print(d_loss_real)
        d_loss=(d_loss_real+d_loss_fake)/2

        fake_x, fake_y = generate_fake_samples(batch_size)
        g_loss=combined.train_on_batch([fake_x,fake_y],np.ones(shape=(batch_size,)))
        print(f"epoch={epoch} : d_loss={d_loss} and g_loss={g_loss}")
        if epoch%100==0:
            save(epoch)

training(1001,256)
g_model.save('generator.h5')
d_model.save('discriminator.h5')
combined.save('combined.h5')











