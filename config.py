class Config(object):
    epoch = 5000
    batch_size = 3
    learning_rate = 0.0008

    cuda = True
    gpu_cnt = 4

    async_loading = True
    pin_memory = True

    root_path = '/home/D2019063/MusicGeneration_VAE-torch'
    data_path = 'data/dataset'
    checkpoint_dir = 'model'
    checkpoint_file = 'checkpoint.pth.tar'
    summary_dir = 'board'

    pretraining_step_size = 220
