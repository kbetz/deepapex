import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

#===========#
# ISODETECT #
#===========#

# MLP for each point using shared weights
class SharedMLP(tf.Module):
    def __init__(self, init_dim, sizes, name=None):
        super().__init__(name=name)
        self.convs = []
        self.bnorms = []
        with self.name_scope:
            for size in sizes:
                # Kernel size 1, since point dims are viewed as channels
                self.convs.append(layers.Conv1D(size, 1,
                                                padding="valid",
                                                activation="relu"))
                self.bnorms.append(layers.BatchNormalization(momentum=0.0))
        
    @tf.Module.with_name_scope
    def __call__(self, x, training=False):
        for (cn, bn) in zip(self.convs, self.bnorms):
            x = cn(x)
            x = bn(x, training=training)
        return x
    
# Sequence of Dense Layers
class DenseSeq(tf.Module):
    def __init__(self, sizes, name=None):
        super().__init__(name=name)
        self.dens = []
        self.bnorms = []
        with self.name_scope:
            for size in sizes:
                # TODO: dropout???
                self.dens.append(layers.Dense(size,
                                              activation="relu",
                                              name="densinator"+str(size) ))
                self.bnorms.append(layers.BatchNormalization(momentum=0.0))
        
    @tf.Module.with_name_scope
    def __call__(self, x, training=False):
        for (dn, bn) in zip(self.dens, self.bnorms):
            x = dn(x)
            x = bn(x, training=training)
        #for i in range(len(self.dens)):
        #    x = self.dens[i](x)
        #    x = self.bnorms[i](x, training=training)
        return x
    
# Regularizer for matrix output of TNet
#@tf.object
class OrthoReg(tf.keras.regularizers.Regularizer):
    def __init__(self, mat_dim, l2reg=0.001):
        self.mat_dim = mat_dim
        self.l2reg = l2reg
        self.eye = tf.eye(mat_dim)
        
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.mat_dim, self.mat_dim))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.mat_dim, self.mat_dim))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    def get_config(self):
        return {"mat_dim": self.mat_dim,
                "l2reg"  : self.l2reg}
    
# Transformation matrix for points/features
class TNet(tf.Module):
    def __init__(self, num_points, mat_dim, att_size=0, name=None):
        super().__init__(name=name)
        self.mat_dim = mat_dim
        self.smlp   = None
        self.mpool  = None
        self.denseq = None
        self.getmat = None
        with self.name_scope:
            self.smlp = SharedMLP(mat_dim, [16, 32])
            self.mpool = layers.MaxPool1D(num_points)
            self.mpool_att = layers.MaxPool1D(num_points*(att_size+1))
            self.denseq = DenseSeq([32, 16])
            self.batchnorm = tf.keras.layers.BatchNormalization()
            self.getmat = layers.Dense(mat_dim * mat_dim,
                                       kernel_initializer="zeros",
                                       bias_initializer=tf.keras.initializers.Constant(np.eye(mat_dim).flatten()),
                                       activity_regularizer=OrthoReg(mat_dim),
                                       name = "final_densination")

    @tf.Module.with_name_scope
    def __call__(self, x, x_att, training=False):
        if x_att is not None:
            x = self.smlp(tf.concat((x, x_att), axis=-2), training)
            x = self.mpool_att(x)
        else:
            x = self.smlp(x, training)
            x = self.mpool(x)
        x = self.batchnorm(x)
        x = self.denseq(x, training)
        x = self.getmat(x)
        return tf.reshape(x, (-1, self.mat_dim, self.mat_dim))


# Attention mechanism
class DANet(tf.Module):
    def __init__(self, point_dim, name=None):
        super().__init__(name=name)
    
    def __call__(self, x):
        return x

class IsoDetect(tf.keras.Model):
    def __init__(self, num_points, att_size, point_dim, num_scores, name=None):
        super().__init__(name=name)
        self.num_points = num_points
        self.tnet1 = TNet(num_points, point_dim, att_size, name="TNet_module_1")
        self.smlp1 = SharedMLP(point_dim, [16,16])
        
        self.tnet2 = TNet(num_points, 16, name="TNet_module_1")
        self.smlp2 = SharedMLP(16, [16,32])
        
        self.mpool = layers.MaxPool1D(num_points)
        self.smlp3 = SharedMLP(64, [256, 128, 64, 16])
        
        self.result = None
        if num_scores > 1:
            self.result = layers.Conv1D(num_scores, 1, padding="same", activation="softmax")
        else:
            self.result = layers.Dense(1, activation='sigmoid')

    def call(self, x_with_att, training=False):
        #if training:
        x = x_with_att[:, :self.num_points]
        x_att = x_with_att[:, self.num_points:]
        #else:
        #    x = x_with_att
        #    x_att = None
            
        tmat1 = self.tnet1(x, x_att, training)
        x = tf.linalg.matmul(x, tmat1)
        x = self.smlp1(x, training)
        
        tmat2 = self.tnet2(x, None, training)
        x = tf.linalg.matmul(x, tmat2)
        x = self.smlp2(x, training)

        mp = self.mpool(x)
        mp = tf.repeat(mp, self.num_points, axis=1)
        x = tf.concat((x, mp), axis=-1)
        x = self.smlp3(x, training)
        
        #TODO: maybe return_logits?
        x = self.result(x)
        
        #if training:
        #    print("train")
        return x
    
# Loss function
@tf.function(experimental_relax_shapes=True)
def weighted_crossentropy(y_true, y_pred, epsilon=1e-7):
    abs_amts = tf.reduce_sum(y_true, axis=0)
    rel_amts = 1 - abs_amts/tf.reduce_sum(abs_amts)
    
    y_pred /= tf.reduce_sum(y_pred, axis=-1,keepdims=True)
    y_clip = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    #w = tf.reduce_sum((rel_amts) * y_true, axis=1)
    #print("----")
    #print(np.sum(w<0.5))
    #print(abs_amts)
    #print(rel_amts)
    #print(1-rel_amts)
    #print("----")
    
    xents = -tf.reduce_sum(y_true * (rel_amts) * tf.math.log(y_clip), -1)
    
    return tf.reduce_mean(xents)

class AddOns:
    custom = {"OrthoReg"             : OrthoReg,
              "weighted_crossentropy": weighted_crossentropy}

    def get_callbacks(log_path):
        return [tf.keras.callbacks.CSVLogger(log_path, append=True),
                tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=3)]

def save_model(model, path):
    model.save(path, save_format="tf", custom_objects=AddOns.custom)
    
def load_model(path):
    return tf.keras.models.load_model(path, custom_objects=AddOns.custom)
    
def get_model(num_points, att_size, point_dim, num_scores):
    isodetect = IsoDetect(num_points, att_size, point_dim, num_scores)
    isodetect.compile(loss      = tf.keras.losses.CategoricalCrossentropy(),#weighted_crossentropy,
                      metrics   = tf.keras.metrics.CategoricalAccuracy(),
                      optimizer = tf.keras.optimizers.Nadam())
    
    return isodetect
