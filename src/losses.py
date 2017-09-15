from keras import backend as K

def lessOneAccuracy(y_true, y_pred):
    shape = K.shape(y_true)
    h = K.reshape(shape[1], (1,1))
    w = K.reshape(shape[2], (1,1))
    denom = 1 / K.cast(K.reshape(K.dot(h, w), (1,1)), dtype = 'float32')
    return K.dot(K.reshape(K.sum(K.cast(K.less_equal(K.abs(y_true - y_pred), 1), dtype = 'float32')), (1,1)), denom)

def lessThreeAccuracy(y_true, y_pred):
    shape = K.shape(y_true)
    h = K.reshape(shape[1], (1,1))
    w = K.reshape(shape[2], (1,1))
    denom = K.dot(h, w)
    denom = 1 / K.cast(K.reshape(K.dot(h, w), (1,1)), dtype = 'float32')
    return K.dot(K.reshape(K.sum(K.cast(K.less_equal(K.abs(y_true - y_pred), 3), dtype = 'float32')), (1,1)), denom)
