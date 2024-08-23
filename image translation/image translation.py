import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Concatenate, LeakyReLU, BatchNormalization, \
    Activation, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras import Model
# from keras.image_utils import load_img,img_to_array
from keras.utils.image_utils import load_img, img_to_array
import os
from keras.initializers.initializers import RandomNormal

# not written kernel initializer yet

def discriminator(img_shape):
    img = Input(shape=img_shape)
    translated = Input(shape=img_shape)
    inp = Concatenate()([img, translated])
    init=RandomNormal(stddev=0.02)

    layer = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same',kernel_initializer=init)(inp)
    layer = LeakyReLU(alpha=0.2)(layer)
    # 128,128,64

    layer = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same',kernel_initializer=init)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    # 64,64,128

    layer = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same',kernel_initializer=init)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    # 32,32,256

    # layer = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same',kernel_initializer=init)(layer)
    # layer = BatchNormalization()(layer)
    # layer = LeakyReLU(alpha=0.2)(layer)
    # # 16,16,512
    #
    # layer = Conv2D(512, kernel_size=(4, 4), padding='same',kernel_initializer=init)(layer)
    # layer = BatchNormalization()(layer)
    # layer = LeakyReLU(alpha=0.2)(layer)
    # # 16,16,512

    layer = Conv2D(1, kernel_size=(4, 4), padding='same',kernel_initializer=init)(layer)
    # 32,32,1
    layer = Activation('sigmoid')(layer)

    model = Model([img, translated], layer)
    optimizer = Adam(0.0015, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def encoder_block(units, prev_layer, batch_normalization=True, ):
    init=RandomNormal(stddev=0.02)
    # layer = Conv2D(filters=units, kernel_size=(4, 4), strides=(2, 2), padding='same',kernel_initializer=init)(prev_layer)
    layer = Conv2D(filters=units, kernel_size=(8,8), strides=(4,4), padding='same', kernel_initializer=init)(prev_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    return layer


def decoder_block(units, inp, concat_layer, dropout=True):
    init=RandomNormal(stddev=0.02)
    layer = Conv2DTranspose(units, kernel_size=(8,8), strides=(4,4), padding='same',kernel_initializer=init)(inp)
    layer = BatchNormalization()(layer)
    # if dropout:
    #     layer=Dropout(0.5)(layer)
    layer = Concatenate()([layer, concat_layer])
    layer = Activation('relu')(layer)
    return layer


def generator(img_shape):
    # img = Input(shape=img_shape)
    # # 256,256,3
    # layer1 = encoder_block(64, img, batch_normalization=False)
    # # 128,128,64
    # layer2 = encoder_block(128, layer1)
    # # 64,64,128
    # layer3 = encoder_block(256, layer2)
    # # 32,32,256
    # layer4 = encoder_block(512, layer3)
    # # 16,16,512
    # layer5 = encoder_block(512, layer4)
    # # 8,8,512
    # layer6 = encoder_block(512, layer5)
    # # 4,4,512
    # b = Conv2D(512, kernel_size=(4, 4), activation='relu', strides=(2, 2), padding='same')(layer6)
    # # 1,1,512

    img=Input(shape=img_shape)
    #256,256,3
    layer1=encoder_block(64,img,batch_normalization=False)
    #64,64,64
    layer2=encoder_block(128,layer1)
    #16,16,128
    layer3=encoder_block(256,layer2)
    #4,4,256
    b=Conv2D(512,kernel_size=(4,4),activation='relu')(layer3)
    #1,1,512

    # d1 = decoder_block(512, b, layer6)
    # # 2,2,1024
    # d2 = decoder_block(512, d1, layer5)
    # # 4,4,1024
    # d3 = decoder_block(512, d2, layer4)
    # # 8,8,1024
    # d4 = decoder_block(512, d3, layer3)
    # # 16,16,1024
    # d5 = decoder_block(256, d4, layer2)
    # # 32,32,512
    # d6 = decoder_block(128, d5, layer1)
    # # 64,64,256
    # d8 = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same')(d6)
    # # 256,256,3
    # final = Activation('tanh')(d8)

    d1=decoder_block(256,b,layer3)
    #4,4,512
    d2=decoder_block(128,d1,layer2)
    #16,16,256
    d3=decoder_block(64,d2,layer1)
    #64,64,128
    d4=Conv2DTranspose(3,kernel_size=(8,8),strides=(4,4),padding='same')(d3)
    #256,256,3
    final=Activation('tanh')(d4)

    model = Model(img, final)
    return model


img_shape = (256, 256, 3)
d_model = discriminator(img_shape)
g_model = generator(img_shape)


def gan(d_model, g_model, img_shape):
    d_model.trainable = False
    image = Input(shape=img_shape)
    gen_image = g_model(image)
    output = d_model([image, gen_image])

    model = Model(image, [output, gen_image])
    optimizer = Adam(0.0015, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, 85])
    return model


combined = gan(d_model, g_model, img_shape)


def get_data():
    folder_path = 'maps/train/'
    real_images = []
    map_images = []
    count = 0
    for filename in os.listdir(folder_path):
        if count >= 100:
            break
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = load_img(image_path, color_mode='rgb', target_size=(256, 512))
            image = img_to_array(image)
            real_img = image[:, :256]
            real_images.append(real_img)
            map_img = image[:, 256:]
            map_images.append(map_img)
        count += 1
    return np.array(real_images), np.array(map_images)


real_images, target_images = get_data()
real_images = (real_images - 127.5) / 127.5
target_images = (target_images - 127.5) / 127.5


def get_image(batch_size, real_images, target_images):
    x = np.random.randint(0, real_images.shape[0], batch_size)
    img_real = real_images[x]
    img_trans = target_images[x]
    return img_real, img_trans
    # return real_images[[0]], target_images[[0]]


# def display_img(g_model):
#   try_image,try_target=get_image(1,real_images,target_images)
#   gen_image=g_model.predict(try_image)
#   try_image=(try_image+1.)/2.
#   try_target=(try_target+1.)/2.
#   gen_image=(gen_image+1.)/2.

#   fig,(ax1,ax2,ax3)= plt.subplots(1,3)
#   ax1.imshow(try_image,cmap='rgb',vmin=0,vmax=1)
#   ax2.imshow(try_target,cmap='rgb',vmin=0,vmax=1)
#   ax3.imshow(gen_image,cmap='rgb',vmin=0,vmax=1)
#   plt.show()

def display_img(g_model,epoch):
    try_image, try_target = get_image(1, real_images, target_images)
    gen_image = g_model.predict(try_image)

    # Rescale pixel values from [-1, 1] to [0, 1]
    try_image = (try_image + 1.) / 2.
    try_target = (try_target + 1.) / 2.
    gen_image = (gen_image + 1.) / 2.

    # Remove the extra dimension
    try_image = try_image.squeeze()
    try_target = try_target.squeeze()
    gen_image = gen_image.squeeze()

    # Plot the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(try_image)
    ax1.set_title("Input Image")
    ax1.axis('off')

    ax2.imshow(try_target)
    ax2.set_title("Target Image")
    ax2.axis('off')

    ax3.imshow(gen_image)
    ax3.set_title("Generated Image")
    ax3.axis('off')
    fig.savefig(f"image_translation/img{epoch}")
    # plt.show()


def train(epochs, batch_size, d_model, g_model, combined):
    try:
        for epoch in range(epochs):
            img_real, img_trans = get_image(int(batch_size / 2), real_images, target_images)
            print(img_real.shape)
            gen_trans = g_model.predict(img_real)
            loss_real = d_model.train_on_batch([img_real, img_trans], np.ones(shape=(img_real.shape[0], 32,32, 1)))
            loss_fake = d_model.train_on_batch([img_real, gen_trans], np.zeros(shape=(img_real.shape[0], 32,32, 1)))
            d_loss = (loss_fake + loss_real) / 2

            real, target = get_image(batch_size, real_images, target_images)
            print(real.shape)
            g_loss = combined.train_on_batch(real, [np.ones(shape=(batch_size, 32,32, 1)), target])
            print(f"epoch={epoch} : d_loss={d_loss} g_loss={g_loss}")

            if epoch % 10 == 0:
                display_img(g_model,epoch)
    except:
        g_model.save('generator_translation.h5')
        d_model.save('discriminator_translation.h5')
        combined.save('combined_translation.h5')

train(501, 64, d_model, g_model, combined)
g_model.save('generator_translation.h5')
d_model.save('discriminator_translation.h5')
combined.save('combined_translation.h5')


















