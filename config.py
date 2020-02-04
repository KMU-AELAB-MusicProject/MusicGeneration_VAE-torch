class Config(object):
    epoch = 1000
    batch_size = 32
    learning_rate = 0.0001

    cuda = True
    gpu_device = '0'

    async_loading = True
    pin_memory = True

    root_path = '/home/algorithm/'
    data_path = 'datasets/data'
    checkpoint_dir = 'model'
    checkpoint_file = 'checkpoint.pth.tar'
    summary_dir = 'board'
