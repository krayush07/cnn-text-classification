class ParamsClass():
    def __init__(self, mode):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.01
        self.max_grad_norm = 5
        self.max_epoch = 2
        self.max_max_epoch = self.max_epoch

        if(mode == 'TR'):
            self.keep_prob = 0.7
        else:
            self.keep_prob = 1.0

        self.lr_decay = 0.5

        self.enable_shuffle = False
        self.enable_checkpoint = False

        if(mode == 'TE'):
            self.enable_shuffle = False

        self.MAX_CONTEXT_SEQ_LENGTH = 80
        self.MAX_UTT_SEQ_LENGTH = 60
        self.NUM_CONTEXT = 2
        self.CONTEXT_SEQ_HIDDEN_DIM = 200
        self.UTT_SEQ_HIDDEN_DIM = 200
        self.EMB_DIM = 300

        self.batch_size = 32
        self.vocab_size = 30
        self.is_word_trainable = True


        self.use_unknown_word = True
        self.use_random_initializer = False


        self.indices = None
        self.num_instances = None


        ''' PARAMS FOR CONV BLOCK '''
        self.num_filters = [64]
        self.filter_width = [[2,3,5]]
        self.conv_activation = 'RELU'
        self.conv_padding = 'SAME'

        self.pool_width = [2]
        self.pool_stride = [1]
        self.pool_padding = 'VALID'
        self.pool_option = 'MAX'
