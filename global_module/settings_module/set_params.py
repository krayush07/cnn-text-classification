class ParamsClass():
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.03
        self.max_grad_norm = 5
        self.max_epoch = 70
        self.max_max_epoch = 100

        if (mode == 'TR'):
            self.keep_prob = 0.4
        else:
            self.keep_prob = 1.0

        self.lr_decay = 0.99

        self.enable_shuffle = False
        self.enable_checkpoint = False
        self.all_lowercase = True

        if (mode == 'TE'):
            self.enable_shuffle = False

        self.REG_CONSTANT = 0.001
        self.MAX_SEQ_LEN = 160
        self.EMB_DIM = 300

        self.batch_size = 256
        self.vocab_size = 30
        self.is_word_trainable = False

        self.use_unknown_word = False
        self.use_random_initializer = False

        self.indices = None
        self.num_instances = None
        self.num_classes = None
        self.sampling_threshold = 3

        ''' PARAMS FOR CONV BLOCK '''
        self.num_filters = [64]
        self.filter_width = [[2, 3, 4, 5]]
        self.conv_activation = 'RELU'
        self.conv_padding = 'VALID'

        self.pool_width = [10]
        self.pool_stride = [1]
        self.pool_padding = 'VALID'
        self.pool_option = 'MAX'
        self.if_pool_max = True # if pool width is equal to convoluted matrix