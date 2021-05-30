import tensorflow as tf 
from keras.layers import Conv1D, Input, Activation, Flatten, Dense, Conv2D
from keras.layers import BatchNormalization, add , Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tcn import TCN


class TCN_Model(Model):
    def __init__(self, savePath, patience):
        super(TCN_Model, self).__init__()
        self.savePath = savePath
        self.patience = patience  
        
        self.conv1 = 
        
    def save_model(self, lookback_window):
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)    
            
        model_path = self.savePath + "_LookBack"+ str(lookback_window) +"epoch_" + "{epoch:04d} -- val_loss_{val_loss: .4f}.hdf5"
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)
        cb_early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience)
        
#         return cb_checkpoint, cb_early_stopping
        
        
        
        
        
        
#  #Residual block :: https://roadcom.tistory.com/95
# def ResBlock(x,filters,kernel_size,dilation_rate):
#     r=Conv2D(filters,kernel_size=kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #first convolution
#     r=Conv2D(filters,kernel_size=kernel_size,padding='same',dilation_rate=dilation_rate)(r) #Second convolution
#     if x.shape[-1]==filters:
#         # Shortcut 의 channel 과 main path 의 channel 이 일치할 경우 단순 add 연산만 진행하는 블록 = identity block
#         shortcut = x  # identity block 
#     else: 
#         # Shortcut 의 channel 과 main path 의 channel 이 다를 경우 shortcut path 를 적절히 변환
#         # 즉, projection 을 통해 channel 을 맞춰주는 작업이(projection shortcut) 추가되기에 이를 convolution block 이라함
#         shortcut=Conv2D(filters,kernel_size=kernel_size,padding='same')(x) 
#     o=add([r,shortcut])
#     o=Activation('relu')(o) 
#     return o
 
#  #Sequence Model
# def TCN(optimizer='adam'):
#     kernel_size = (3,3)
#     input_shape =  (lookback_window, 3, 1) # (8,3, 1) = (feature, sliding_window, 1)
    
#     inputs=Input(shape=input_shape)
    
#     x=ResBlock(x = inputs,filters=32,kernel_size=kernel_size,dilation_rate=1)
#     x = Dropout(0.2) (x)
#     x=ResBlock(x,filters=32,kernel_size=kernel_size,dilation_rate=2)
#     x = Dropout(0.4)(x)
#     x=ResBlock(x,filters=16,kernel_size=kernel_size,dilation_rate=4)
    
#     x=Flatten()(x)
#     x=Dense(numActions, activation='softmax')(x)
#     model=Model(inputs=inputs,outputs=x)
         
#     model.summary()
        
#     model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
 
#     return model
