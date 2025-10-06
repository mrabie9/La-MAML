class ModelFactory():
    def __init__(self): pass

    def get_model():
        nm_channels = 112
        channels = 256
        size_of_representation = 512
        size_of_interpreter = 56560
        cin = 2

        return [
            ('conv1_nm', [nm_channels, cin, 5, 1]),
            ('bn1_nm', [nm_channels]),
            ('conv2_nm', [nm_channels, nm_channels, 5, 1]),
            ('bn2_nm', [nm_channels]),
            ('conv3_nm', [nm_channels, nm_channels, 5, 1]),
            ('bn3_nm', [nm_channels]),
            ('nm_to_fc', [size_of_representation, size_of_interpreter]),
        ]
        return [
            ('conv1d_nm', [nm_channels, in_ch, 5, 1, 2]),   # cout, cin, k, stride, pad
            ('bn1d_nm',  [nm_channels]),
            ('conv1d_nm', [nm_channels, nm_channels, 5, 1, 2]),
            ('bn1d_nm',  [nm_channels]),
            ('max_pool1d', [2]),                            # kernel=2 (stride=2)
            ('conv1d_nm', [nm_channels, nm_channels, 5, 1, 2]),
            ('bn1d_nm',  [nm_channels]),
            # ('max_pool1d', [2]),   # add if you want another downsample
            ('nm_to_fc', [size_of_representation, 'AUTO']), # 'AUTO' → we’ll infer input dim
        ]