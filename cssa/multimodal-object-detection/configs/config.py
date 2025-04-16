config = {
    # Training Hyperparameters
    'lr': 2.5e-4,
    'epochs': 50,
    'batch_size': 16,

    # CSSA Module Parameters
    'reduction_factor': 4,
    'cssa_switching_thresh': 2e-3,

    # Model Parameters
    'num_classes': 2,  # Number of classes (including background)

    # Data Parameters
    'train_data_path': 'data/raw/train',
    'val_data_path': 'data/raw/val',
    'processed_data_path': 'data/processed',

    # Checkpoint and Logging
    'checkpoint_interval': 5,
    'checkpoint_path': 'checkpoints/best_checkpoint.pth',

    # Device
    'device': 'cuda',  # 'cuda' or 'cpu'
}
