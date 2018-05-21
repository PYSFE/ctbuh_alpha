import tensorflow as tf
import copy

tf.logging.set_verbosity(tf.logging.INFO)


def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat and batch the samples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""

    if not isinstance(features, dict):
        features = dict(features)

    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def model(train, test, training_steps=10000, model_dir=None, layers=1, neurons_per_layer=2):
    # PREPARATION
    # ============

    # Prepare data
    # ------------

    # split features and labels
    train_feature, train_label = train
    test_feature, test_label = test

    # make a copy of combined data, for later extracting categorical lists
    all_features = copy.copy(train_feature)
    all_features = all_features.append(test_feature, ignore_index=True)

    all_labels = copy.copy(train_label)
    all_labels = all_labels.append(test_label, ignore_index=True)

    # MISC
    # ====

    # print string template
    print_template = '{:<30}{:}'

    # MAKE FEATURE COLUMNS
    # ====================

    # Floors, number integer with unknown string
    floors_numeric = tf.feature_column.numeric_column("floors")
    floors = tf.feature_column.bucketized_column(
        source_column=floors_numeric,
        boundaries=[-1, 0, 15, 30, 45, 60]
    )

    # Primary type, strings
    primary_type_category = list(all_features['primary_type'].drop_duplicates())
    primary_type_category.sort()
    primary_type = tf.feature_column.categorical_column_with_vocabulary_list(
        key='primary_type',
        vocabulary_list=primary_type_category
    )

    # Height, number with unknown string
    height_numeric = tf.feature_column.numeric_column('height')
    height = tf.feature_column.bucketized_column(
        source_column=height_numeric,
        boundaries=[-1, 0, 10, 20, 30, 40, 50, 60]
    )

    # Country, strings
    country_category = list(all_features['country'].drop_duplicates())
    country_category.sort()
    country = tf.feature_column.categorical_column_with_vocabulary_list(
        key='country',
        vocabulary_list=country_category
    )
    # print(country_category)

    # Year built, number integer with unknown
    year_built_numeric = tf.feature_column.numeric_column(key='year_built')
    year_built = tf.feature_column.bucketized_column(
        source_column=year_built_numeric,
        boundaries=[-1, 0, 1960, 1970, 1980, 1990, 2000, 2010]
    )

    # Age at incident, number with unknown
    # age_at_incident_numeric = tf.feature_column.numeric_column(key='age_at_incident')
    # age_at_incident = tf.feature_column.bucketized_column(
    #     source_column=age_at_incident_numeric,
    #     boundaries=[0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]
    # )

    # Construction, strings
    construction_category = list(all_features['construction'].drop_duplicates())
    construction_category.sort()
    construction = tf.feature_column.categorical_column_with_vocabulary_list(
        key='construction',
        vocabulary_list=construction_category
    )

    # Facade type, string
    facade_type_category = list(all_features['facade_type'].drop_duplicates())
    facade_type_category.sort()
    facade_type = tf.feature_column.categorical_column_with_vocabulary_list(
        key='facade_type',
        vocabulary_list=facade_type_category
    )

    # Cavity, string
    cavity_category = list(all_features['cavity'].drop_duplicates())
    cavity_category.sort()
    cavity = tf.feature_column.categorical_column_with_vocabulary_list(
        key='cavity',
        vocabulary_list=cavity_category
    )

    # Face material, string
    face_material_category = list(all_features['face_material'].drop_duplicates())
    face_material_category.sort()
    face_material = tf.feature_column.categorical_column_with_vocabulary_list(
        key='face_material',
        vocabulary_list=face_material_category
    )

    # Core material, string
    core_material_category = list(all_features['core_material'].drop_duplicates())
    core_material_category.sort()
    core_material = tf.feature_column.categorical_column_with_vocabulary_list(
        key='core_material',
        vocabulary_list=core_material_category
    )

    # Insulation material, string
    insulation_material_category = list(all_features['insulation_material'].drop_duplicates())
    insulation_material_category.sort()
    insulation_material = tf.feature_column.categorical_column_with_vocabulary_list(
        key='insulation_material',
        vocabulary_list=insulation_material_category
    )

    # Ignition, string
    # ignition_category = list(all_features['ignition'].drop_duplicates())
    # ignition_category.sort()
    # print(ignition_category)
    # ignition = tf.feature_column.categorical_column_with_vocabulary_list(
    #     key='ignition',
    #     vocabulary_list=ignition_category
    # )

    # Ignition floor, numeric with unknown
    # ignition_floor_numeric = tf.feature_column.numeric_column(key='ignition_floor')
    # ignition_floor = tf.feature_column.bucketized_column(
    #     source_column=ignition_floor_numeric,
    #     boundaries=[0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]
    # )

    # Fatalities, THIS IS PRIMARY LABEL

    # Injuries, THIS IS SECONDARY LABEL

    # Sprinklers, String
    sprinklers_category = list(all_features['sprinklers'].drop_duplicates())
    sprinklers_category.sort()
    sprinklers = tf.feature_column.categorical_column_with_vocabulary_list(
        key='sprinklers',
        vocabulary_list=sprinklers_category
    )

    # Wind, string
    # wind_category = list(train_feature['wind'].drop_duplicates())
    # wind_category.sort()
    # print(wind_category)
    # wind = tf.feature_column.categorical_column_with_vocabulary_list(
    #     key='wind',
    #     vocabulary_list=wind_category
    # )

    # Falling material, string
    # falling_material_category = list(all_features['falling_material'].drop_duplicates())
    # falling_material_category.sort()
    # print(falling_material_category)
    # falling_material = tf.feature_column.categorical_column_with_vocabulary_list(
    #     key='falling_material',
    #     vocabulary_list=falling_material_category
    # )

    # Under construction, string (YES, NO)
    under_construction_category = list(all_features['under_construction'].drop_duplicates())
    under_construction_category.sort()
    under_construction = tf.feature_column.categorical_column_with_vocabulary_list(
        key='under_construction',
        vocabulary_list=under_construction_category
    )

    my_feature_columns = [floors,
                          floors_numeric,
                          tf.feature_column.indicator_column(primary_type),
                          height,
                          height_numeric,
                          tf.feature_column.indicator_column(country),
                          year_built,
                          year_built_numeric,
                          # age_at_incident,
                          tf.feature_column.indicator_column(construction),
                          tf.feature_column.indicator_column(facade_type),
                          tf.feature_column.indicator_column(cavity),
                          tf.feature_column.indicator_column(face_material),
                          tf.feature_column.indicator_column(core_material),
                          tf.feature_column.indicator_column(insulation_material),
                          # tf.feature_column.indicator_column(ignition),
                          # ignition_floor,
                          tf.feature_column.indicator_column(sprinklers),
                          # tf.feature_column.indicator_column(wind),
                          # tf.feature_column.indicator_column(falling_material),
                          tf.feature_column.indicator_column(under_construction),
                          ]

    # my_feature_columns = [tf.feature_column.indicator_column(i) for i in my_feature_columns]

    # MODEL BODY
    # ==========

    hidden_units = [neurons_per_layer] * layers

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        model_dir=model_dir,
        hidden_units=hidden_units,
        n_classes=12
    )

    # TRAINING
    # ========

    classifier.train(
        input_fn=lambda: train_input_fn(train_feature, train_label, 5),
        steps=training_steps
    )

    # VALIDATION
    # ==========

    eval_result_train = classifier.evaluate(
        input_fn=lambda: eval_input_fn(train_feature, train_label, batch_size=5)
    )

    eval_result_test = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_feature, test_label, batch_size=5)
    )

    # BASELINE ACCURACY
    # =================

    # baseline_classifier = tf.estimator.BaselineClassifier(n_classes=12)
    #
    # baseline_classifier.train(
    #     input_fn=lambda: train_input_fn(all_features, all_labels, batch_size=5),
    # )
    #
    # baseline_loss = baseline_classifier.evaluate(
    #     input_fn=lambda: eval_input_fn(all_features, all_labels, batch_size=5)
    # )

    # PREDICTION
    # ==========

    # expected = test_label
    # predict_x = {}
    # for key in test_feature.keys():
    #     predict_x[key] = test_feature[key]
    #
    # predictions = classifier.predict(
    #     input_fn=lambda: eval_input_fn(predict_x, batch_size=10))
    #
    # template = ('Prediction is "{}" ({:.1f}%), expected "{}"')
    #
    # prediction_list = []
    # probability_list = []
    # expected_list = []

    # df.from_dict(predict_x).to_csv('test_sample.csv')

    # PRINT OUTPUT
    # ============

    # Print model configurations
    # --------------------------

    str_all_labels = list(all_labels.drop_duplicates())
    str_all_labels = [str(i) for i in str_all_labels]
    print(print_template.format('hidden_units:', hidden_units))
    print(print_template.format('Layers:', layers))
    print(print_template.format('Neurons:', neurons_per_layer))
    print(print_template.format('Training data:', train_feature.shape[0]))
    print(print_template.format('Testing data:', test_feature.shape[0]))
    print(print_template.format('Steps:', training_steps))
    print(print_template.format('Total categories:', len(str_all_labels)))
    print(print_template.format('Categories:', ', '.join(str_all_labels)))
    print('Train set accuracy:           {accuracy:0.3f}'.format(**eval_result_train))
    print('Test set accuracy:            {accuracy:0.3f}\n'.format(**eval_result_test))

    return eval_result_train['accuracy'], eval_result_test['accuracy']
