import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl


def make_cat_calibrators(input, lattice_size, conf):
    calibrator = tfl.layers.CategoricalCalibration(
        num_buckets=2,
        output_min=0.0,
        output_max=lattice_size - 1.0,

        # Initializes all outputs to (output_min + output_max) / 2.0.
        kernel_initializer='constant',
        name='{}_calib'.format(conf),
    )(input)

    return calibrator


def make_calibrators(data, lattice_size, units, name=None, monotonicity='none'):
    try:
        min_ = data.min()
        max_ = data.max()
    except:
        min_ = 0
        max_ = 1
        if data == 'Uy': max_ = 2
        if data =='Z': max_=2
    calibrator = tfl.layers.PWLCalibration(
        # Easiest way to specify them is to uniformly cover entire input range by using numpy.linspace().
        # For Categorical Variables see tfl.layers.CategoricalCalibration.
        input_keypoints=np.linspace(
            min_, max_, num=5),
        dtype=tf.float32,
        monotonicity=monotonicity,
        units=units,
        # Output range must correspond to expected lattice input range.
        output_min=0.0,
        output_max=lattice_size - 1.0,
        name=name
    )
    return calibrator

def make_multiple_calibrators(confounders, params):
    i = 2
    model_inputs = []
    lattice_inputs = []
    for  conf in confounders.columns:
        _input = tf.keras.layers.Input(shape=[1], name=conf)
        model_inputs.append(_input)
        calib = make_calibrators(confounders[conf].values, params.lattice_sizes[i],
                                   params.z_calib_units,
                                   monotonicity=params.z_monotonicity[i - 2], name='{}_calib'.format(conf))(_input)
        i += 1
        lattice_inputs.append(calib)

    return model_inputs, lattice_inputs


def Twin_Net_with_Z_A(treatment, uy, confounders, params):
    lattice_sizes = params.lattice_sizes
    model_inputs = []

    a_input = tf.keras.layers.Input(shape=[1], name='A')
    model_inputs.append(a_input)
    a_calibrator = make_calibrators(treatment.values, lattice_sizes[0], params.calib_units, monotonicity='increasing',
                                    name='a_calib')(a_input)

    a_prime_input = tf.keras.layers.Input(shape=[1], name='A_prime')
    model_inputs.append(a_prime_input)
    a_prime_calibrator = make_calibrators(treatment.values, lattice_sizes[0], params.calib_units,
                                          monotonicity='increasing',
                                          name='a_prime_calib')(a_prime_input)

    uy_input = tf.keras.layers.Input(shape=[1], name='Uy')
    model_inputs.append(uy_input)
    uy_calibrator = make_calibrators(uy, lattice_sizes[1], params.hidden_dims, monotonicity=params.uy_monotonicity
                                     , name='uy_name')(uy_input)

    if params.multiple_confounders:
        inputs_, z_calibrator = make_multiple_calibrators(confounders, params)
        _ = [model_inputs.append(i) for i in inputs_]
        layer_z = z_calibrator

    else:
        confounders = confounders.values

        z_input = tf.keras.layers.Input(shape=(params.len_conf,), name='Z')
        model_inputs.append(z_input)
        z_calibrator = make_calibrators(confounders, lattice_sizes[2], params.z_calib_units,
                                        monotonicity=params.z_monotonicity[0])(z_input)
        z_calibrator = [z_calibrator]
        # Z layer
        if params.z_layer == 'linear':
            z_calibrator = tf.tile(z_calibrator[..., tf.newaxis], (1, 1, params.z_calib_units))
            layer_z = tfl.layers.Linear(params.z_calib_units, units=params.z_calib_units,
                                        )(z_calibrator)
            layer_z = [layer_z]
        else:
            layer_z = z_calibrator

    # Uy layer
    if params.uy_layer == 'linear':
        layer_uy = tfl.layers.Linear(1, units=params.calib_units, monotonicities=params.uy_monotonicity)(uy_calibrator)
    else:
        layer_uy = uy_calibrator


    conc_a_z = tf.keras.layers.Concatenate()([a_calibrator, *layer_z])
    conc_a_prime_z = tf.keras.layers.Concatenate()([a_prime_calibrator, *layer_z])
    layer_a_z = tfl.layers.Linear( np.sum([i.shape[-1] for i in layer_z]) +lattice_sizes[0], units=1, monotonicities='increasing')(conc_a_z)
    layer_a_z_prime = tfl.layers.Linear(np.sum([i.shape[-1] for i in layer_z]) +lattice_sizes[0], units=1, monotonicities='increasing')(conc_a_prime_z)


    if params.concats:
        conc_input = tf.keras.layers.Concatenate(axis=1)([layer_a_z, layer_uy, *layer_z])
        conc_input_prime  = tf.keras.layers.Concatenate(axis=1)([layer_a_z_prime, layer_uy, *layer_z])
        lattice_y = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z]) + lattice_sizes[1]+1,
            units=params.lattice_units,
            monotonicities= 'increasing',
            name='Y',
        )(conc_input)

        lattice_y_prime = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z])+ lattice_sizes[1] + 1,
            units=params.lattice_units,
            monotonicities='increasing',
            name='Y_prime',
        )(conc_input_prime)
    else:
        lattice_y = tfl.layers.Lattice(
            lattice_sizes=[2,*lattice_sizes[1:]],
            units=params.lattice_units,
            monotonicities=[
                'increasing', params.uy_monotonicity, *params.z_monotonicity
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y',
        )([layer_a_z, layer_uy, *layer_z])

        lattice_y_prime = tfl.layers.Lattice(
            lattice_sizes=[2,*lattice_sizes[1:]],
            units=params.lattice_units,
            monotonicities=[
                'increasing', params.uy_monotonicity, *params.z_monotonicity
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y_prime',
        )([layer_a_z_prime, layer_uy, *layer_z])


    if params.end_activation == 'calib':
        lattice_y = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0.0, 1.0, 5),
            name='output_calib',
        )(
            lattice_y)
    elif params.end_activation == 'softmax':
        lattice_y = tf.keras.layers.Softmax()(lattice_y)
        lattice_y_prime = tf.keras.layers.Softmax()(lattice_y_prime)
    elif params.end_activation == 'sigmoid':
        lattice_y = tf.nn.sigmoid(lattice_y)
        lattice_y_prime = tf.nn.sigmoid(lattice_y_prime)

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[lattice_y, lattice_y_prime])

    return model

def Twin_Net(treatment, uy, confounders, params):
    lattice_sizes = params.lattice_sizes
    model_inputs = []

    a_input = tf.keras.layers.Input(shape=[1], name='A')
    model_inputs.append(a_input)
    a_calibrator = make_calibrators(treatment.values, lattice_sizes[0], params.calib_units, monotonicity='increasing',
                                    name='a_calib')(a_input)

    a_prime_input = tf.keras.layers.Input(shape=[1], name='A_prime')
    model_inputs.append(a_prime_input)
    a_prime_calibrator = make_calibrators(treatment.values, lattice_sizes[0], params.calib_units,
                                          monotonicity='increasing',
                                          name='a_prime_calib')(a_prime_input)

    uy_input = tf.keras.layers.Input(shape=[1], name='Uy')
    model_inputs.append(uy_input)
    uy_calibrator = make_calibrators(uy, lattice_sizes[1], params.hidden_dims, monotonicity=params.uy_monotonicity
                                     , name='uy_name')(uy_input)

    if params.multiple_confounders:
        inputs_, z_calibrator = make_multiple_calibrators(confounders, params)
        _ = [model_inputs.append(i) for i in inputs_]
        layer_z = z_calibrator

    else:
        confounders = confounders.values

        z_input = tf.keras.layers.Input(shape=(params.len_conf,), name='Z')
        model_inputs.append(z_input)
        z_calibrator = make_calibrators(confounders, lattice_sizes[2], params.z_calib_units,
                                        monotonicity=params.z_monotonicity[0])(z_input)
        # Z layer
        if params.z_layer == 'linear':
            z_calibrator = tf.tile(z_calibrator[..., tf.newaxis], (1, 1, params.z_calib_units))
            layer_z = tfl.layers.Linear(params.z_calib_units, units=params.z_calib_units,
                                        )(z_calibrator)
            layer_z = [layer_z]
        else:
            layer_z = [z_calibrator]

    # Uy layer
    if params.uy_layer == 'linear':
        layer_uy = tfl.layers.Linear(1, units=params.calib_units, monotonicities=params.uy_monotonicity)(uy_calibrator)
    else:
        layer_uy = uy_calibrator

    if params.layer == 'lattice':
        lattice_y = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            units=params.lattice_units,
            monotonicities=[
                'increasing', params.uy_monotonicity, *params.z_monotonicity
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y',
        )([a_calibrator, layer_uy, *layer_z])

        lattice_y_prime = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            units=params.lattice_units,
            monotonicities=[
                'increasing', params.uy_monotonicity, *params.z_monotonicity
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y_prime',
        )([a_prime_calibrator, layer_uy, *layer_z])
    else:
        conc_input = tf.keras.layers.Concatenate(axis=1)([a_calibrator, layer_uy, *layer_z])

        conc_input_prime = tf.keras.layers.Concatenate(axis=1)([a_prime_calibrator, layer_uy, *layer_z])

        if params.lattice_units >1:
            conc_input = tf.expand_dims(conc_input,1)
            conc_input = tf.tile(conc_input, [1,2,1], name=None)
            conc_input_prime = tf.expand_dims(conc_input_prime, 1)
            conc_input_prime = tf.tile(conc_input_prime, [1, 2, 1], name=None)

        lattice_y = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z]) + lattice_sizes[0] + lattice_sizes[1],
            units=params.lattice_units,
            monotonicities=
            'increasing'
            ,

            name='Y',
        )(conc_input)

        lattice_y_prime = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z]) + lattice_sizes[0] + lattice_sizes[1],
            units=params.lattice_units,
            monotonicities=
            'increasing',
            name='Y_prime',
        )(conc_input_prime)


    if params.end_activation == 'calib':
        lattice_y = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0.0, 1.0, 5),
            name='output_calib',
        )(
            lattice_y)
    elif params.end_activation == 'softmax':
        lattice_y = tf.keras.layers.Softmax()(lattice_y)
        lattice_y_prime = tf.keras.layers.Softmax()(lattice_y_prime)
    elif params.end_activation == 'sigmoid':
        lattice_y = tf.nn.sigmoid(lattice_y)
        lattice_y_prime = tf.nn.sigmoid(lattice_y_prime)

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[lattice_y, lattice_y_prime])

    return model



def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def DiceBCELoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors

    scce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    BCE = scce(targets, inputs)
    dc = dice_loss(targets,inputs)
    Dice_BCE = 0.5*BCE + 1.5*dc

    return Dice_BCE


def class_loss(class_weight):
  """Returns a loss function for a specific class weight tensor

  Params:
    class_weight: 1-D constant tensor of class weights

  Returns:
    A loss function where each loss is scaled according to the observed class"""
  class_weight = tf.dtypes.cast(class_weight, tf.float32)
  # scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  scce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  def loss(y_obs, y_pred):
    dc = dice_loss(y_obs,y_pred)


    # y_pred = tf.dtypes.cast(y_pred, tf.int32)
    y_obs = tf.dtypes.cast(y_obs, tf.int32)

    hothot = tf.one_hot(tf.reshape(y_obs, [-1]), depth=class_weight.shape[0])
    weight = tf.math.multiply(class_weight, hothot)
    weight = tf.reduce_sum(weight, axis=-1)
    losses = scce(y_obs,
                  y_pred,
                  sample_weight=weight)
    return losses + dc

  return loss

