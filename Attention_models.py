"""
Christopher Brook
10603660
MSc Health Data Science

"""
# imports for this project
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.applications import ResNet152V2
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, multiply, Lambda, AvgPool2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

"""
The part of the following in my own but certain parts have been used for an online source, as referenced below:

Mader, K., 2018. Cardiomegaly Pretrained-VGG16. [online] Kaggle.com. Available at: <https://www.kaggle.com/kmader/cardiomegaly-pretrained-vgg16> [Accessed 20 August 2020].

"""


# importing the CSV containing the location of the diseases, finding within the images and population data.
xray_ds = pd.read_csv('C:/MSCDISS/FULL_Data_Entry_2017_updated.csv')

# creating a list of the classes
disease_list = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis",
 "Hernia","Infiltration","Mass", "No Finding","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]

# looping through a creating a attention map per disease.
for disease in disease_list:
    xray_ds[disease] = xray_ds['Finding Labels'].map(lambda x: disease  in x)

    train_df, valid_df = train_test_split(xray_ds, test_size = 0.30)

    # setting the image sizes
    IMG_SIZE = (224, 224)

    # configuring the alterations to the images.
    img_gen = ImageDataGenerator(rescale=1. / 255,
                                 preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
                                 brightness_range=[0.2, 0.7], fill_mode='nearest',
                                 zoom_range=0.125, samplewise_std_normalization=True)


    # creating the training and validation batches class
    def flow_from_dataframe(img_data_gen, in_df, location_col, y_col, **dflow_args):
        df_gen = img_data_gen.flow_from_dataframe(in_df,
                                                  x_col=location_col,
                                                  y_col=y_col,
                                         class_mode = 'raw',
                                        **dflow_args)
        return df_gen


    # creating the training and validation batches
    train_gen = flow_from_dataframe(img_gen, train_df,
                                 location_col = 'location',
                                y_col = disease ,
                                target_size = IMG_SIZE,
                                 color_mode = 'rgb',
                                batch_size = 32)
    
    valid_gen = flow_from_dataframe(img_gen, valid_df,
                                 location_col = 'location',
                                y_col = disease ,
                                target_size = IMG_SIZE,
                                 color_mode = 'rgb',
                                batch_size = 256) # we can use much larger batches for evaluation

    # used a fixed dataset for evaluating the algorithm
    test_X, test_Y = next(flow_from_dataframe(img_gen,
                                   valid_df,
                                 location_col = 'location',
                                y_col = disease ,
                                target_size = IMG_SIZE,
                                 color_mode = 'rgb',
                                batch_size = 256)) # one big batch

    t_x, t_y = next(train_gen)


    base_model = ResNet152V2(input_shape =  t_x.shape[1:],
                                  include_top = False, weights = 'imagenet')
    base_model.trainable = False
    # (Mader, 2018)
    pt_features = Input(base_model.get_output_shape_at(0)[1:], name = 'feature_input')
    pt_depth = base_model.get_output_shape_at(0)[-1]
    bn_features = BatchNormalization(name='Features_BN')(pt_features)
    # (Mader, 2018)
    attn_layer = Conv2D(180, kernel_size = (1,1), padding = 'same', activation = 'elu')(bn_features)
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='elu')(attn_layer)
    attn_layer = Conv2D(32, kernel_size=(1, 1), padding='same', activation='elu')(attn_layer)
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='elu')(attn_layer)
    attn_layer = AvgPool2D((2, 2), strides=(1, 1), padding='same')(attn_layer)
    attn_layer = Conv2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid',
                       name='AttentionMap2D')(attn_layer)

    #(Mader, 2018)
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', name='UpscaleAttention',
                   activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False

    # (Mader, 2018)
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # (Mader, 2018)
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    # (Mader, 2018)
    gap_dr = Dropout(0.5)(gap)
    dr_steps = Dropout(0.5)(Dense(128, activation = 'elu')(gap_dr))
    out_layer = Dense(1, activation = 'sigmoid')(dr_steps)
    
    attn_model = Model(inputs = [pt_features], outputs = [out_layer], name = 'attention_model')

    attn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    model = Sequential(name = 'combined_model')
    base_model.trainable = False
    model.add(base_model)
    model.add(attn_model)
    # (Mader, 2018)
    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy')
    weight_path = "{}_weights.best.hdf5".format(disease + '_attn')
    # saving model weights
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)
    # setting the batch size
    train_gen.batch_size = 24
    # fitting the model/
    model.fit_generator(train_gen,
                          validation_data = (test_X, test_Y),
                           steps_per_epoch=train_gen.n//train_gen.batch_size,
                          epochs = 2,
                        callbacks=[checkpoint],
                          workers = 3)

    model.load_weights(weight_path)

    # (Mader, 2018)
    # creation and output of the attention model.
    for attn_layer in attn_model.layers:
        c_shape = attn_layer.get_output_shape_at(0)
        if len(c_shape)==4:
            if c_shape[-1]==1:
                print(attn_layer)
                break
    # (Mader, 2018)
    rand_idx = np.random.choice(range(len(test_X)), size = 6)
    attn_func = K.function(inputs = [attn_model.get_input_at(0), K.learning_phase()],
               outputs = [attn_layer.get_output_at(0)])
    # (Mader, 2018)
    fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]
    for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
        cur_img = test_X[c_idx:(c_idx+1)]
        cur_features = base_model.predict(cur_img)
        attn_img = attn_func([cur_features, 0])[0]
        img_ax.imshow(cur_img[0,:,:,0], cmap = 'bone')
        attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis',
                       vmin = 0, vmax = 1,
                       interpolation = 'lanczos')
        real_label = test_Y[c_idx]
        img_ax.set_title(disease +'\nClass:%s' % (real_label))
        pred_confidence = model.predict(cur_img)[0]
        attn_ax.set_title(disease + 'Map\nPred:%2.1f%%' % (100*pred_confidence[0]))
    path = 'C:/Users/chris/PycharmProjects/MSC_diss_final/code/Attention_output/'
    attention_name = disease + '_map.png'
    filename_loc = os.path.join(path, attention_name)
    fig.savefig(filename_loc, dpi = 400)

    print("done")

