from tensorboard.plugins.hparams import api as hp

HP_FINAL_OUTPUT_CHANNEL_SIZE = hp.HParam('output_channel_size', hp.Discrete([64, 128]))
HP_DENSE_LAYER_SIZES = hp.HParam(
    'dense_layer_sizes', 
    hp.Discrete(
        [
           "[500, 250, 125, 25]",
           "[500, 250, 125, 75, 25]",
           "[400, 200, 100, 50]",
           "[800, 400, 200, 100, 50]",
        ]
    )
)
METRIC_ACCURACY = 'accuracy'
