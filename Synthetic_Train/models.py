import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl

def make_calibrators(data, lattice_size,params):
    try:
        min_ = data.min()
        max_ = data.max()
    except:
        min_ = 0
        max_ = 1
        if data == 'Uy': max_ = 2

    calibrator = tfl.layers.PWLCalibration(
        # Easiest way to specify them is to uniformly cover entire input range by using numpy.linspace().
        # For Categorical Variables see tfl.layers.CategoricalCalibration.
        input_keypoints=np.linspace(
            min_, max_, num=5),
        dtype=tf.float32,
        units=params.calib_units,
        # Output range must correspond to expected lattice input range.
        output_min=0.0,
        output_max=lattice_size - 1.0
    )
    return calibrator


def TwinNet(features, params):

    lattice_sizes = params.lattice_sizes
    model_inputs = []

    xs_input = tf.keras.layers.Input(shape=[1], name='Xs')
    model_inputs.append(xs_input)
    x_calibrator = make_calibrators(features['X'],lattice_sizes[0],params)(xs_input)


    uy_input = tf.keras.layers.Input(shape=[1], name='Uy')
    model_inputs.append(uy_input)
    uy_calibrator = make_calibrators(features['Uy'], lattice_sizes[1],params)(uy_input)
    if params.uy_layer == 'linear':
        layer_uy = tfl.layers.Linear(1, units=params.hidden_dims, monotonicities=params.uy_monotonicity)(uy_calibrator)
    elif params.uy_layer == 'lattice':
        layer_uy = tfl.layers.Lattice([lattice_sizes[1]],units=1,monotonicities=[params.uy_monotonicity],
                                                output_min=0, output_max=2,)
    else:
        layer_uy = uy_calibrator



    xs_prime_input = tf.keras.layers.Input(shape=[1], name='Xs_prime')
    model_inputs.append(xs_prime_input)
    x_prime_calibrator = make_calibrators(features['X'], lattice_sizes[0],params)(xs_prime_input)

    if params.calib_units == 1:
        lattice_sizes = lattice_sizes[3:]
    else:
        lattice_sizes = [lattice_sizes[3],params.calib_units]


    if params.end_activation == 'sigmoid':
        lattice_y = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            monotonicities=[
                'increasing', params.uy_monotonicity,
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y_',
        )([x_calibrator,layer_uy])

        lattice_y = tf.nn.sigmoid(lattice_y,'Y')

        lattice_y_prime = tfl.layers.Lattice(
                lattice_sizes=lattice_sizes,
                monotonicities=[
                    'increasing', params.uy_monotonicity,
                ],
                output_min=0.0,
                output_max=1.0,
                name='Y_prime_',
            )([x_prime_calibrator,layer_uy])

        lattice_y_prime = tf.nn.sigmoid(lattice_y_prime, 'Y_prime')
    else:
        lattice_y = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            monotonicities=[
                'increasing', params.uy_monotonicity,
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y',
        )([x_calibrator, layer_uy])


        lattice_y_prime = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            monotonicities=[
                'increasing', params.uy_monotonicity,
            ],
            output_min=0.0,
            output_max=1.0,
            name='Y_prime',
        )([x_prime_calibrator, layer_uy])

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[lattice_y, lattice_y_prime])

    return model


def SingleNet(features, params):
    lattice_sizes = params.lattice_sizes
    model_inputs = []

    xs_input = tf.keras.layers.Input(shape=[1], name='Xs')
    model_inputs.append(xs_input)
    x_calibrator = make_calibrators(features['X'],lattice_sizes[0],params)(xs_input)


    uy_input = tf.keras.layers.Input(shape=[1], name='Uy')
    model_inputs.append(uy_input)
    uy_calibrator = make_calibrators(features['Uy'], lattice_sizes[1],params)(uy_input)
    if params.uy_layer == 'linear':
        layer_uy = tfl.layers.Linear(1, units=params.hidden_dims, monotonicities=params.uy_monotonicity)(uy_calibrator)
    elif params.uy_layer == 'lattice':
        layer_uy = tfl.layers.Lattice([lattice_sizes[1]],units=1,monotonicities=[params.uy_monotonicity],
                                                output_min=0, output_max=2,)
    else:
        layer_uy = uy_calibrator

    lattice_y = tfl.layers.Lattice(
        lattice_sizes=lattice_sizes[3:],
        monotonicities=[
            'increasing', params.uy_monotonicity,
        ],
        output_min=0.0,
        output_max=1.0,
        name='Y',
    )([x_calibrator, layer_uy])

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=lattice_y)

    return model


def custom_loss(y_actual,y_pred):
    # print(y_actual.shape)
    mse =  tf.keras.losses.mean_squared_error
    factual_loss = 0.1*mse(y_actual,y_pred)
    return factual_loss
