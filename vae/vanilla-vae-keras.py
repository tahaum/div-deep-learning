import os
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ----------- Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ----------- Set variables
lr = 0.0005
max_epochs = 50
x_size = 784 # Dimension of 28x28 images flattened
h_size = 256 # Intermediate layer dimension
z_size = 2  # Encoding dimension
batch_size = 100
x_train = np.reshape(x_train, (x_train.shape[0], x_size))/255 # Reshape and normalize (0, 255 range for each pixel)
x_test = np.reshape(x_test, (x_test.shape[0], x_size))/255

# ----------- Define model
x = Input(batch_shape = (batch_size, x_size))
h = Dense(h_size, input_shape=(batch_size, x_size), activation='relu', kernel_initializer='he_normal',
    bias_initializer='zeros')(x)
z_mean = Dense(z_size, input_shape=(batch_size, h_size), activation=None, kernel_initializer='glorot_normal',
    bias_initializer='zeros')(h)
z_logvar = Dense(z_size, input_shape=(batch_size, h_size), activation=None, kernel_initializer='glorot_normal',
    bias_initializer='zeros')(h)

def sample_z(args):
    z_mean, z_logvar = args
    return z_mean + K.exp(z_logvar/2)*np.random.randn(batch_size, z_size)
z = Lambda(sample_z)([z_mean, z_logvar])

decoder_1 = Dense(h_size, input_shape=(batch_size, z_size), activation='relu', kernel_initializer='he_normal',
    bias_initializer='zeros')
decoder_2 = Dense(x_size, input_shape=(batch_size, h_size), activation='sigmoid', kernel_initializer='glorot_normal',
    bias_initializer='zeros')
h = decoder_1(z)
y = decoder_2(h)

def vae_loss(y_true, y_pred):
    reconstruction_loss = K.sum(binary_crossentropy(y_true, y_pred), axis=-1)
    KL_div = -0.5*np.sum(1 + z_logvar - np.square(z_mean) - K.exp(z_logvar))
    return K.mean(reconstruction_loss + KL_div)

model = Model(x, y)
rmsprop = RMSprop(lr=lr)
model.compile(loss=vae_loss, optimizer=rmsprop)
print(model.summary())

# ----------- Training
if not os.path.exists('output/'):
    os.makedirs('output')

checkpoint_path = 'output/checkpoint-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1,
    save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1)
callbacks_list = [checkpoint, reduce_lr]
model.fit(x_train, x_train, shuffle=True, epochs=max_epochs, batch_size=batch_size,
    validation_data=(x_test, x_test), callbacks=callbacks_list, verbose=0)

# ----------- Generate and plot
z_gen = Input(batch_shape=(None, z_size))
h_gen = decoder_1(z_gen)
y_gen = decoder_2(h_gen)
generator = Model(z_gen, y_gen)
print(generator.summary())

nr_imgs = 100
dim = int(np.sqrt(nr_imgs))
step = 3/dim
x = np.r_[-1.5:1.5:step]
y = np.r_[-1.5:1.5:step]
imgs = list()
for i in range(dim):
    for j in range(dim):
        samp = np.array([[x[i], y[j]]])
        imgs.append(255*generator.predict(samp))

imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

fig = plt.figure(figsize=(dim, dim))
gs = gridspec.GridSpec(dim, dim)
gs.update(wspace=0.05, hspace=0.05)

for i, img in enumerate(imgs):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(img, cmap='Greys_r')
    
plt.savefig('latent_space.png', bbox_inches='tight')
plt.show()