# DF model used for non-defended dataset
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization,SeparableConv1D,GlobalAveragePooling1D,Attention,Input,Add,Softmax,ELU,ReLU
from keras.layers.core import Activation, Flatten, Dense, Dropout
import tensorflow as tf
from keras import backend as K
from keras.models import Model
def SqueezeAndExcitation(inputs, ratio=2):
    input_shape = K.int_shape(inputs)
    print(input_shape)
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    excitation = tf.keras.layers.Dense(units = input_shape[-1]//ratio, kernel_initializer='he_normal',activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(units = input_shape[-1],activation='sigmoid')(excitation)
    #excitation = tf.reshape(excitation, [-1, 1, input_shape[-1]])
    scale = tf.keras.layers.Multiply()([inputs, excitation])
    return scale

class DFNet:
    @staticmethod
    def build_BASE(input_shape, classes):
        input_data = Input(shape=input_shape)
        #Block1

        model = Conv1D(filters=32, kernel_size=5,strides=2,padding='same')(input_data)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = Conv1D(filters=64, kernel_size=5,strides=2,padding='same')(model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = MaxPooling1D(pool_size=5, strides=2)(model)

        residual_model = Conv1D(filters=128,kernel_size=1,strides=1,padding='same')(model)
        model = SqueezeAndExcitation(model)
        #Block2*4
        
        model = SeparableConv1D(filters=128,kernel_size=5,strides=1,padding='same')(model)
        
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)

        model = SeparableConv1D(filters=128,kernel_size=5,strides=1,padding='same')(model)
        # model = Add()([residual_model,model])
        model = tf.add(model,residual_model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = MaxPooling1D(pool_size=5, strides=2)(model)

        residual_model = Conv1D(filters=256,kernel_size=1,strides=1,padding='same')(model)

        model = SeparableConv1D(filters=256,kernel_size=5,strides=1,padding='same')(model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)

        model = SeparableConv1D(filters=256,kernel_size=5,strides=1,padding='same')(model)
        # model = add([residual_model,model])
        model = tf.add(model,residual_model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = MaxPooling1D(pool_size=5, strides=2)(model)

        residual_model = Conv1D(filters=728,kernel_size=1,strides=1,padding='same')(model)
        

        model = SeparableConv1D(filters=728,kernel_size=5,strides=1,padding='same')(model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)

        model = SeparableConv1D(filters=728,kernel_size=5,strides=1,padding='same')(model)
        # model = Add()([residual_model,model])
        model = tf.add(model,residual_model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = MaxPooling1D(pool_size=5, strides=2)(model)

        residual_model = Conv1D(filters=1024,kernel_size=1,strides=1,padding='same')(model)
        

        model = SeparableConv1D(filters=728,kernel_size=5,strides=1,padding='same')(model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)

        model = SeparableConv1D(filters=1024,kernel_size=5,strides=1,padding='same')(model)
        # model = Add()([residual_model,model])
        model = tf.add(model,residual_model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = MaxPooling1D(pool_size=5, strides=2)(model)


        #Block3
        model = SeparableConv1D(filters=1536,kernel_size=5,strides=1,padding='same')(model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)
        model = SeparableConv1D(filters=2048,kernel_size=5,strides=1,padding='same')(model)
        model = BatchNormalization(axis=-1)(model)
        model = ELU(alpha=1.0)(model)

        model = SqueezeAndExcitation(model)
        
        model = GlobalAveragePooling1D()(model)

        model = Dense(classes,activation='softmax')(model)
        model = Model(inputs=input_data, outputs=model)
        return model
    

    @staticmethod
    def build_DAE(input_shape):
        input_data = Input(shape=input_shape)
        model = Flatten()(input_data)
        model = Dropout(0.1)(model)
        model = Dense(750)(model)
        model = ReLU()(model)
        model = Dense(500)(model)
        model = ReLU(name='get_feature')(model)
        model = Dense(750)(model)
        model = ReLU()(model)
        model = Dense(5000)(model)
        # model = Softmax()(model)
        model = Model(inputs=input_data, outputs=model)
        return model



