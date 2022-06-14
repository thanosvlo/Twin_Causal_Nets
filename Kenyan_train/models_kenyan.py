import sys

sys.path.append("..")
sys.path.append("../..")

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl


def make_calibrators(data, lattice_size, units, name=None, monotonicity='none'):
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
        monotonicity=monotonicity,
        units=units,
        # Output range must correspond to expected lattice input range.
        output_min=0.0,
        output_max=lattice_size - 1.0,
        name=name
    )
    return calibrator


def make_multiple_calibrators(confounders, params):
    model_inputs = []
    lattice_inputs = []
    i = 2

    if 'c13_c_child_gender' in confounders:
        gender_input = tf.keras.layers.Input(shape=[1], name='c13_c_child_gender')
        model_inputs.append(gender_input)
        gender_calibrator = tfl.layers.CategoricalCalibration(
            num_buckets=2,
            output_min=0.0,
            output_max=params.lattice_sizes[i] - 1.0,

            # Initializes all outputs to (output_min + output_max) / 2.0.
            kernel_initializer='constant',
            name='gender_calib',
        )(gender_input)
        i += 1
        lattice_inputs.append(gender_calibrator)

    if 'base_age' in confounders:
        age_input = tf.keras.layers.Input(shape=[1], name='base_age')
        model_inputs.append(age_input)
        age_calib = make_calibrators(confounders['base_age'].values, params.lattice_sizes[i], params.z_calib_units,
                                     monotonicity=params.z_monotonicity[i - 2],
                                     name='age_calib')(age_input)
        i += 1
        lattice_inputs.append(age_calib)

    if 'momeduc_orig' in confounders:
        momed_input = tf.keras.layers.Input(shape=[1], name='momeduc_orig')
        model_inputs.append(momed_input)
        momed_calib = make_calibrators(confounders['momeduc_orig'].values, params.lattice_sizes[i],
                                       params.z_calib_units,
                                       monotonicity=params.z_monotonicity[i - 2], name='momeduc_calib', )(momed_input)
        i += 1
        lattice_inputs.append(momed_calib)

    if 'splnecmpn_base' in confounders:
        splnecmpn_base_input = tf.keras.layers.Input(shape=[1], name='splnecmpn_base')
        model_inputs.append(splnecmpn_base_input)
        splnecmpn_base_calib = make_calibrators(confounders['splnecmpn_base'].values, params.lattice_sizes[i],
                                                params.z_calib_units,
                                                monotonicity=params.z_monotonicity[i - 2], name='splnecmpn_base_calib')(
            splnecmpn_base_input)
        i += 1
        lattice_inputs.append(splnecmpn_base_calib)

    if 'e1_iron_roof_base' in confounders:
        e1_iron_roof_base_input = tf.keras.layers.Input(shape=[1], name='e1_iron_roof_base')
        model_inputs.append(e1_iron_roof_base_input)
        e1_iron_roof_base_calib = make_calibrators(confounders['e1_iron_roof_base'].values, params.lattice_sizes[i],
                                                   params.z_calib_units,
                                                   monotonicity=params.z_monotonicity[i - 2],
                                                   name='e1_iron_roof_base_calib')(e1_iron_roof_base_input)
        i += 1
        lattice_inputs.append(e1_iron_roof_base_calib)

    if 'hygiene_know_base' in confounders:
        hygiene_know_base_input = tf.keras.layers.Input(shape=[1], name='hygiene_know_base')
        model_inputs.append(hygiene_know_base_input)
        hygiene_know_base_calib = make_calibrators(confounders['hygiene_know_base'].values, params.lattice_sizes[i],
                                                   params.z_calib_units,
                                                   monotonicity=params.z_monotonicity[i - 2],
                                                   name='hygiene_know_base_calib')(hygiene_know_base_input)
        i += 1
        lattice_inputs.append(hygiene_know_base_calib)

    if 'latrine_density_base' in confounders:
        latrine_density_base_input = tf.keras.layers.Input(shape=[1], name='latrine_density_base')
        model_inputs.append(latrine_density_base_input)
        latrine_density_base_calib = make_calibrators(confounders['latrine_density_base'].values,
                                                      params.lattice_sizes[i],
                                                      params.z_calib_units,
                                                      monotonicity=params.z_monotonicity[i - 2],
                                                      name='latrine_density_base_calib')(latrine_density_base_input)
        i += 1
        lattice_inputs.append(latrine_density_base_calib)

    if 'numkids_base' in confounders:
        numkids_base_input = tf.keras.layers.Input(shape=[1], name='numkids_base')
        model_inputs.append(numkids_base_input)
        numkids_base_calib = make_calibrators(confounders['numkids_base'].values, params.lattice_sizes[i],
                                              params.z_calib_units,
                                              monotonicity=params.z_monotonicity[i - 2], name='numkids_base_calib')(
            numkids_base_input)
        i += 1
        lattice_inputs.append(numkids_base_calib)

    return model_inputs, lattice_inputs


def Single_Twin_Net_Kenyan(treatment, uy, confounders, params):
    lattice_sizes = params.lattice_sizes
    model_inputs = []
    lattice_inputs = []

    a_input = tf.keras.layers.Input(shape=[1], name='A')
    model_inputs.append(a_input)
    a_calibrator = make_calibrators(treatment.values, lattice_sizes[0], params.calib_units, monotonicity='increasing',
                                    name='a_calib')(a_input)
    lattice_inputs.append(a_calibrator)

    uy_input = tf.keras.layers.Input(shape=[1], name='Uy')
    model_inputs.append(uy_input)
    uy_calibrator = make_calibrators(uy, lattice_sizes[1], params.hidden_dims, monotonicity=params.uy_monotonicity
                                     , name='uy_name')(uy_input)
    lattice_inputs.append(uy_calibrator)

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
    if params.end_activation == 'calib':
        lattice_y = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0.0, 1.0, 5),
            name='output_calib',
        )(
            lattice_y)
    elif params.end_activation == 'softmax':
        lattice_y = tf.keras.layers.Softmax()(lattice_y)

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=lattice_y)

    return model


def Twin_Net_Kenyan(treatment, uy, confounders, params):
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
    if params.layer =='lattice':
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
        lattice_y = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z]) + lattice_sizes[0] +lattice_sizes[1] ,
            units=params.lattice_units,
            monotonicities=
            'increasing'
            ,

            name='Y',
        )(conc_input)

        lattice_y_prime = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z]) + lattice_sizes[0] +lattice_sizes[1] ,
            units=params.lattice_units,
            monotonicities=
            'increasing'
            ,

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


def Twin_Net_Kenyan_with_Propensity(treatment, uy, confounders, params):
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

    lattice_A_prime = tfl.layers.Lattice(
        lattice_sizes=lattice_sizes[2:],
        units=params.lattice_units,
        monotonicities=[
            *params.z_monotonicity
        ],
        output_min=0.0,
        output_max=1.0,
        name='A_out_prime',
    )([*layer_z])

    lattice_A = tfl.layers.Lattice(
        lattice_sizes=lattice_sizes[2:],
        units=params.lattice_units,
        monotonicities=[
            *params.z_monotonicity
        ],
        output_min=0.0,
        output_max=1.0,
        name='A_out',
    )([*layer_z])

    if params.end_activation == 'calib':
        lattice_y = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0.0, 1.0, 5),
            name='output_calib',
        )(
            lattice_y)
    elif params.end_activation == 'softmax':
        lattice_y = tf.keras.layers.Softmax()(lattice_y)
    elif params.end_activation == 'sigmoid':
        lattice_y = tf.nn.sigmoid(lattice_y)
        lattice_y_prime = tf.nn.sigmoid(lattice_y_prime)

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[lattice_y, lattice_y_prime, lattice_A, lattice_A_prime])

    return model


def Twin_Net_Kenyan_with_Z_A(treatment, uy, confounders, params):
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
            monotonicities=
                'increasing'
            ,


            name='Y',
        )(conc_input)

        lattice_y_prime = tfl.layers.Linear(
            np.sum([i.shape[-1] for i in layer_z])+ lattice_sizes[1] + 1,
            units=params.lattice_units,
            monotonicities=
            'increasing'
            ,

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
    elif params.end_activation == 'sigmoid':
        lattice_y = tf.nn.sigmoid(lattice_y)
        lattice_y_prime = tf.nn.sigmoid(lattice_y_prime)

    model = tf.keras.models.Model(
        inputs=model_inputs,
        outputs=[lattice_y, lattice_y_prime])

    return model
