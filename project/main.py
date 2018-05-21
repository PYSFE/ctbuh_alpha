from func_ana import nested


if __name__ == '__main__':

    # SETTINGS
    # ========

    #
    model_dir = None  # './tf_models/real_model_3'

    layers_list = [1]
    neurons_per_layer_list = [2]
    training_steps_list = [500]
    training_items_list = [42]

    # Number of models to be trained

    n_runs = 1

    nested(
        model_dir=model_dir,
        n_runs=n_runs,
        layers_list=layers_list,
        neurons_per_layer_list=neurons_per_layer_list,
        training_steps_list=training_steps_list,
        training_items_list=training_items_list,
    )
