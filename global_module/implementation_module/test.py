import tensorflow as tf
import numpy as np
import sys
import time

from global_module.settings_module import set_params, set_dir, set_dict
from global_module.implementation_module import model
from global_module.implementation_module import reader


def run_epoch(session, eval_op, model_obj, dict_obj, verbose=False):
    epoch_combined_loss = 0.0
    total_correct = 0.0
    total_instances = 0.0
    print('\nrun epoch')

    output_file = open(set_dir.Directory('TE').test_cost_path, 'w')

    params = model_obj.params
    dir_obj = model_obj.dir_obj
    data_filename = dir_obj.data_filename
    label_filename = dir_obj.label_filename

    for step, (input_seq_arr, label_arr) \
            in enumerate(reader.data_iterator(params, data_filename, label_filename, model_obj.params.indices, dict_obj)):
        feed_dict = {}
        feed_dict[model_obj.word_input] = input_seq_arr
        feed_dict[model_obj.label] = label_arr

        loss, prediction, probabilities, _ = session.run([model_obj.loss,
                                                          model_obj.prediction,
                                                          model_obj.probabilities,
                                                          eval_op],
                                                         feed_dict=feed_dict)

        total_correct += np.sum(prediction == label_arr)
        total_instances += params.batch_size
        epoch_combined_loss += loss

        for each_pred in prediction:
            output_file.write(str(each_pred) + '\n')

    print 'CE loss: %.4f, Accuracy: %.4f' % ((epoch_combined_loss), (total_correct / total_instances) * 100)

    return epoch_combined_loss


def getLength(fileName):
    print('Reading :', fileName)
    dataFile = open(fileName, 'r')
    count = 0
    for _ in dataFile:
        count += 1
    dataFile.close()
    return count, np.arange(count)


def init_test():
    mode_train, mode_test = 'TR', 'TE'

    dict_obj = set_dict.Dictionary()

    # train object
    params_train = set_params.ParamsClass(mode=mode_train)
    dir_train = set_dir.Directory(mode_train)
    params_train.num_classes = len(dict_obj.label_dict)

    # test object
    params_test = set_params.ParamsClass(mode=mode_test)
    dir_test = set_dir.Directory(mode_test)
    params_test.num_instances, params_test.indices = getLength(dir_test.data_filename)
    params_test.batch_size = 1
    params_test.num_classes = len(dict_obj.label_dict)

    min_loss = sys.float_info.max

    word_emb_path = dir_train.word_embedding
    word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
    params_train.vocab_size = params_test.vocab_size = len(word_emb_matrix)

    print('***** INITIALIZING TF GRAPH *****')

    session = tf.Session()

    with tf.name_scope('train'):
        with tf.variable_scope("model", reuse=None):
            test_obj = model.CNNClassification(params_test, dir_test)

    model_saver = tf.train.Saver()
    print('Loading model ...')
    model_saver.restore(session, set_dir.Directory('TE').test_model)

    print('**** MODEL LOADED ****\n')

    return session, test_obj


def run_test(session, test_obj, dict_obj):
    start_time = time.time()

    print("Starting test computation\n")
    test_loss = run_epoch(session, tf.no_op(), test_obj, dict_obj)

    curr_time = time.time()
    print('1 epoch run takes ' + str(((curr_time - start_time) / 60)) + ' minutes.')

# def main():
#     session, test_obj = init_test()
#     dict_obj = set_dict.Dictionary()
#     run_test(session, test_obj, dict_obj)
#
#
# if __name__ == "__main__":
#     main()

