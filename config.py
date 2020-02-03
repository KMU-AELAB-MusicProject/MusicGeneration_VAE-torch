class Config(object):
    epoch = 1000
    batch_size = 32
    learning_rate = 0.0001

    cuda = True
    gpu_device = 0

    async_loading = True

    root_path = '/home/algorithm/'
    checkpoint_file = 'checkpoint.pth.tar'
    summary_dir = ''
