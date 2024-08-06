import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU, BatchNormalization,Conv2DTranspose
from keras.optimizers import Adam
from keras import Input, Model, Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt

img_shape = (28, 28, 1)


def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=2,padding='valid'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=2,padding='valid'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=64,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))


    image = Input(shape=img_shape)
    valid = model(image)
    return Model(image, valid)
    # model = Sequential()
    #
    # model.add(Flatten(input_shape=img_shape))
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(256))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.summary()
    #
    # img = Input(shape=img_shape)
    # validity = model(img)
    #
    # return Model(img, validity)





discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5) , metrics=['accuracy'])


def build_generator():
    noise_shape = (100,)

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)  # Generated image

    return Model(noise, img)



generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

(x_train, _), (_, _) = mnist.load_data()


x_train=(x_train-127.5)/127.5
# print(x_train[0])
# x_train=x_train[:10000]
x_train = x_train
x_train = np.expand_dims(x_train, axis=3)
print(x_train[0])

# for declaration of combined model
in_test = Input(shape=(100,))
gen_images_test = generator(in_test)
print("tp")
print(gen_images_test.shape)
discriminator.trainable = False
output_test = discriminator(gen_images_test)
combined = Model(in_test, output_test)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5) )


def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100,))
    gen_imgs = generator.predict(noise)
    # print(gen_imgs[:2])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray', vmin=0, vmax=1)
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images4/mnist_%d.png" % epoch)
    # plt.show()
    plt.close()


def train(x_train, epochs,batch_size):
    half_size = int(batch_size / 2)
    full_size = batch_size

    for epoch in range(epochs):
        # generate images from generator
        inn = np.random.normal(0, 1, size=(half_size, 100))  # mean 0 std 1
        gen_images = generator.predict(inn)
        print(gen_images.shape)

        # train discriminator
        indexes = np.random.randint(0, x_train.shape[0], half_size)
        train_images = x_train[indexes]

        y_valid = np.ones((half_size,))
        y_not = np.zeros((half_size,))
        d_loss_real = discriminator.train_on_batch(train_images, y_valid)
        d_loss_fake = discriminator.train_on_batch(gen_images, y_not)
        d_loss = np.average(d_loss_real + d_loss_fake)

        # train combined model
        input_combined = np.random.normal(0, 1,size=(full_size, 100))
        output_combined = np.ones(shape=(full_size,))
        combined_loss = combined.train_on_batch(input_combined, output_combined)

        print(f"epoch no {epoch}: loss={d_loss} combined_loss={combined_loss}")
        if epoch % 100 == 0 or epoch==999:
            save_imgs(epoch)



train(x_train, 10000,128)

