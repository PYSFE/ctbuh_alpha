import gc
import numpy as np
from pandas import DataFrame as df
from func_data import load_data
from func_model import model


def nested(
        model_dir=None,
        n_runs=10,
        layers_list=(2,),
        neurons_per_layer_list=(10,),
        training_steps_list=(2000,),
        training_items_list=(42,)
):
    result_dict = {
        'layers': [],
        'neurons_per_layer': [],
        'training_step': [],
        'training_data': [],
        'test accuracy': [],
        'train accuracy': [],
    }

    train_individual_accuracy_all = []
    test_individual_accuracy_all = []

    count_sim = 0
    total_sim = len(layers_list) * len(neurons_per_layer_list) * len(training_steps_list) * len(
        training_items_list) * n_runs - 1

    for layers in layers_list:
        for neurons_per_layer in neurons_per_layer_list:
            for training_steps in training_steps_list:
                for training_items in training_items_list:

                    train_individual_accuracy = []
                    test_individual_accuracy = []

                    for i in np.arange(0, n_runs):
                        print('{}/{}'.format(count_sim, total_sim))
                        count_sim += 1

                        train, test = load_data(n_train=training_items, shuffle=True)

                        train_accuracy, test_accuracy = model(
                            train, test,
                            model_dir=model_dir,
                            training_steps=training_steps,
                            layers=layers,
                            neurons_per_layer=neurons_per_layer
                        )

                        train_individual_accuracy.append(train_accuracy)
                        test_individual_accuracy.append(test_accuracy)

                    train_individual_accuracy_all.append(train_individual_accuracy)
                    test_individual_accuracy_all.append(test_individual_accuracy)

                    result_dict['layers'].append(layers)
                    result_dict['neurons_per_layer'].append(neurons_per_layer)
                    result_dict['training_step'].append(training_steps)
                    result_dict['training_data'].append(training_items)
                    result_dict['test accuracy'].append(np.average(test_individual_accuracy))
                    result_dict['train accuracy'].append(np.average(train_individual_accuracy))

    test_individual_accuracy_all = np.asarray(test_individual_accuracy_all)
    train_individual_accuracy_all = np.asarray(train_individual_accuracy_all)
    np.savetxt('results_test_accuracy4.csv', test_individual_accuracy_all, delimiter=',')
    np.savetxt('results_train_accuracy4.csv', train_individual_accuracy_all, delimiter=',')

    df_result = df.from_dict(result_dict)
    df_result.to_csv('results_summary4.csv')

    # shutil.rmtree(model_dir, ignore_errors=True)

    gc.collect()