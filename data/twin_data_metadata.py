
confounder_monotonicities_1_some = {'anemia': 'none',
                               'cardiac':'none',
                              'lung': 'none',
                              'diabetes': 'none',
                               'gestat10':'none',
                               'herpes':'none',
                               'hydra':'none',
                               'hemo':'none',
       'chyper':'none', 'phyper':'none', 'eclamp':'none', 'incervix':'none', 'pre4000':'none', 'preterm':'none',
                               'renal':'none', 'rh':'none', 'uterine':'none', 'othermr':'none', 'tobacco':'none',
                               'alcohol':'none','cigar6':'none','crace':'none'
                               }

confounder_monotonicities_2 = {'anemia': 'increasing',
                               'cardiac':'increasing',
                              'lung': 'increasing',
                              'diabetes': 'increasing',
                                'gestat10':'increasing',
                               'herpes':'increasing',
                               'hydra':'increasing',
                               'hemo':'increasing',
       'chyper':'increasing', 'phyper':'increasing', 'eclamp':'increasing', 'incervix':'increasing',
                               'pre4000':'increasing', 'preterm':'increasing',
                               'renal':'increasing', 'rh':'increasing', 'uterine':'increasing',
                               'othermr':'increasing', 'tobacco':'increasing',
                               'alcohol':'increasing'
                               }


cov_description = {'adequacy': 'adequacy of care',
 'alcohol': 'risk factor, alcohol use',
 'anemia': 'risk factor, Anemia',
 'birattnd': 'medical person attending birth',
 'birmon': 'birth month Jan-Dec',
 'bord_0': 'birth order of lighter twin',
 'bord_1': 'birth order of heavier twin',
 'brstate': 'state of residence NCHS',
 'brstate_reg': 'US census region of brstate',
 'cardiac': 'risk factor, Cardiac',
 'chyper': 'risk factor, Hypertension, chronic',
 'cigar6': 'risk num of cigarettes /day, quantiled',
 'crace': 'risk race of child',
 'csex': 'risk sex of child',
 'data_year': 'year: 1989, 1990 or 1991',
 'dfageq': 'risk octile age of father',
 'diabetes': 'risk factor, Diabetes',
 'dlivord_min': 'risk number of live births before twins',
 'dmar': 'married',
 'drink5': 'risk num of drinks /week, quantiled',
 'dtotord_min': 'risk total number of births before twins',
 'eclamp': 'risk factor, Eclampsia',
 'feduc6': 'education category',
 'frace': 'dad race',
 'gestat10': 'risk gestation 10 categories',
 'hemo': 'risk factor Hemoglobinopathy',
 'herpes': 'risk factor, Herpes',
 'hydra': 'risk factor Hvdramnios/Oliqohvdramnios',
 'incervix': 'risk factor, Incompetent cervix',
 'infant_id_0': 'infant id of lighter twin in original df',
 'infant_id_1': 'infant id of heavier twin in original df',
 'lung': 'risk factor, Lung',
 'mager8': 'risk mom age',
 'meduc6': 'mom education',
 'mplbir': 'mom place of birth',
 'mplbir_reg': 'US census region of mplbir',
 'mpre5': 'risk trimester prenatal care begun, 4 is none',
 'mrace': 'mom race',
 'nprevistq': 'risk quintile number of prenatal visits',
 'orfath': 'dad hispanic',
 'ormoth': 'mom hispanic',
 'othermr': 'risk factor, Other Medical Risk Factors',
 'phyper': 'risk factor, Hypertension, preqnancy-associated',
 'pldel': 'place of delivery',
 'pre4000': 'risk factor, Previous infant 4000+ grams',
 'preterm': 'risk factor, Previos pre-term or small',
 'renal': 'risk factor, Renal disease',
 'rh': 'risk factor, RH sensitization',
 'stoccfipb': 'state of occurence FIPB',
 'stoccfipb_reg': 'US census region of stoccfipb',
 'tobacco': 'risk factor, tobacco use',
 'uterine': 'risk factor, Uterine bleeding'}




cov_types = {

'Z_0':'none',
    'Z_1':'none',
    'Z_2':'none',
    'Z_3':'none',
    'Z_4':'none',
    'Z_5':'none',
    'Z_6':'none',
    'Z_7':'none',
    'Z_8':'none',
    'Z_9':'none',
    'Z_10':'none',
    'Z_11':'none',
    'Z_12':'none',
    'Z_13':'none',
    'Z_14':'none',
    'Z_15':'none',
    'Z_16':'none',
    'Z_17':'none',

 'adequacy': 'cat',
 'alcohol': 'bin',
 'anemia': 'bin',
 'birattnd': 'cat',
 'birmon': 'cyc',
 'bord': 'bin',
 'brstate': 'cat',
 'brstate_reg': 'cat',
 'cardiac': 'bin',
 'chyper': 'bin',
 'cigar6': 'cat',
 'crace': 'cat',
 'csex': 'bin',
 'data_year': 'cat',
 'dfageq': 'cat',
 'diabetes': 'bin',
 'dlivord_min': 'ord',
 'dmar': 'bin',
 'drink5': 'cat',
 'dtotord_min': 'ord',
 'eclamp': 'bin',
 'feduc6': 'cat',
 'frace': 'cat',
 'gestat10': 'cat',
 'hemo': 'bin',
 'herpes': 'bin',
 'hydra': 'bin',
 'incervix': 'bin',
 'infant_id': 'index do not use',
 'lung': 'bin',
 'mager8': 'cat',
 'meduc6': 'cat',
 'mplbir': 'cat',
 'mpre5': 'cat',
 'mrace': 'cat',
 'nprevistq': 'cat',
 'orfath': 'cat',
 'ormoth': 'cat',
 'othermr': 'bin',
 'phyper': 'bin',
 'pldel': 'cat',
 'pre4000': 'bin',
 'preterm': 'bin',
 'renal': 'bin',
 'rh': 'bin',
 'stoccfipb': 'cat',
 'stoccfipb_reg': 'cat',
 'tobacco': 'bin',
 'uterine': 'bin',

           # 'dtotord': 'none',
           #
           #   'cigar': 'none',
           #   'drink': 'none',
           #   'wtgain': 'none',
           #
           #   'gestat': 'none',
           #   'dmage': 'none',
           #   'dmeduc': 'none',
           #
           #   'resstatb': 'none',
           #   'mpcb': 'none',
           #   'nprevist': 'none',

             'dtotord': 'ord',

             'cigar': 'cat',
             'drink': 'bini',
             'wtgain': 'none',

             'gestat': 'cat',
             'dmage': 'none',
             'dmeduc': 'cat',

             'resstatb': 'none',
             'mpcb': 'none',
             'nprevist': 'none',

             }


confounder_monotonicities_1 = {'adequacy': 'none',
 'alcohol': 'none',
 'anemia': 'none',
 'birattnd': 'none',
 'birmon': 'none',
 'bord_0': 'none',
 'bord_1': 'none',
 'brstate': 'none',
 'brstate_reg': 'none',
 'cardiac': 'none',
 'chyper': 'none',
 'cigar6': 'none',
 'crace': 'none',
 'csex': 'none',
 'data_year': 'none',
 'dfageq': 'none',
 'diabetes': 'none',
 'dlivord_min': 'none',
 'dmar': 'none',
 'drink5': 'none',

 'dtotord_min': 'none',
'dtotord':'none',
 'eclamp': 'none',
 'feduc6': 'none',
 'frace': 'none',
 'gestat10': 'none',
 'hemo': 'none',
 'herpes': 'none',
 'hydra': 'none',
 'incervix': 'none',
 'infant_id_0': 'none',
 'infant_id_1': 'none',
 'lung': 'none',
 'mager8': 'none',
 'meduc6': 'none',
 'mplbir': 'none',
 'mplbir_reg': 'none',
 'mpre5': 'none',
 'mrace': 'none',

 'nprevistq': 'none',
'nprevist':'none',
'resstatb':'none',
 'orfath': 'none',
 'ormoth': 'none',
 'othermr': 'none',
 'phyper': 'none',
 'pldel': 'none',
 'pre4000': 'none',
 'preterm': 'none',
 'renal': 'none',
 'rh': 'none',
 'stoccfipb': 'none',
 'stoccfipb_reg': 'none',
 'tobacco': 'none',
 'uterine': 'none',


'cigar':'none',
'drink':'none',
'wtgain':'none',

'gestat':'none',
'dmage':'none',
'dmeduc':'none',

'mpcb':'none',


                               }
