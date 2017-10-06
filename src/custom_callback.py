from keras.callbacks import Callback
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.layers import Input
class customModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, cost_weight_filepath, linear_output_weight_filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', period=1):
        super(customModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.cost_weight_filepath = cost_weight_filepath
 	self.linear_output_weight_filepath = linear_output_weight_filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    def custom_save_weights(self, overwrite):  
        cost = self.model.layers[-2].output
        cost_model = Model(self.model.input, cost)
        cost_model.save_weights(self.cost_weight_filepath, overwrite)
        if self.linear_output_weight_filepath:
             linear_output = self.model.layers[-1]
             b, m, h, w = K.int_shape(cost)
             linear_input = Input((m, h, w))
             linear_model = Model(linear_input, linear_output(linear_input))
             linear_model.save_weights(self.linear_output_weight_filepath, overwrite)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving weight to %s and %s'
                                  % (epoch, self.monitor, self.best,
                                     current, self.cost_weight_filepath, self.linear_output_weight_filepath))
                        self.best = current
		        self.custom_save_weights(True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s and %s' % (epoch,self.cost_weight_filepath, self.linear_output_weight_filepath))
		self.custom_save_weights(True)
