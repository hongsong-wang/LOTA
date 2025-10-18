import argparse
import os
import torch


class ConfigurationManager:
    def __init__(self):
        self.config_initialized = False

    def define_arguments(self, argument_parser):
        """Define all configuration parameters"""
        # Training hyperparameters
        argument_parser.add_argument('--batchsize', type=int, default=64,
                                     help='Number of samples processed simultaneously')
        argument_parser.add_argument('--choices', default=[1, 1, 1, 1, 1, 1, 1, 1],
                                     help='Dataset selection flags')
        argument_parser.add_argument('--epoch', type=int, default=30,
                                     help='Total training iterations')
        argument_parser.add_argument('--lr', type=float, default=0.0001,
                                     help='Initial learning rate')
        argument_parser.add_argument('--load', type=str, default=None,
                                     help='Path to pre-trained model weights')
        argument_parser.add_argument('--image_root', type=str,
                                     default='/home/hdd1/chengrenxi/GenImage',
                                     help='Root directory for image datasets')
        argument_parser.add_argument('--save_path', type=str,
                                     default='/home/hdd1/chengrenxi/sdv5_thresholding2/',
                                     help='Directory for saving model outputs')
        argument_parser.add_argument('--isPatch', type=bool, default=True,
                                     help='Enable patch processing mode')
        argument_parser.add_argument('--img_height', type=int, default=256,
                                     help='Target image height')
        argument_parser.add_argument('--bit_mode', type=str, default='scaling',
                                     choices=['scaling', 'thresholding'],
                                     help='Bit plane processing method')
        argument_parser.add_argument('--patch_size', type=int, default=32,
                                     help='Dimension for patch extraction')
        argument_parser.add_argument('--patch_mode', type=str, default='max',
                                     choices=['max', 'min', 'random'],
                                     help='Patch selection strategy')
        argument_parser.add_argument('--gpu_id', type=str, default='5',
                                     help='Identifier for GPU device')
        argument_parser.add_argument('--val_batchsize', type=int, default=64,
                                     help='Batch size for validation')
        return argument_parser

    def collect_arguments(self):
        """Gather and process command-line arguments"""
        if not self.config_initialized:
            argument_parser = argparse.ArgumentParser(
                description='Model Training Configuration',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            argument_parser = self.define_arguments(argument_parser)
            self.config_initialized = True

        config, _ = argument_parser.parse_known_args()
        self.argument_parser = argument_parser

        return self.argument_parser.parse_args()

    def display_configuration(self, config):
        """Present configuration details in a structured format"""
        configuration_details = []
        configuration_details.append("╔══════════════════════════════════════════╗")
        configuration_details.append("║          TRAINING CONFIGURATION           ║")
        configuration_details.append("╠══════════════════════════════════════════╣")

        for parameter, value in sorted(vars(config).items()):
            default_value = self.argument_parser.get_default(parameter)
            configuration_details.append(f"║ {parameter:>20}: {str(value):<20}")

        configuration_details.append("╚══════════════════════════════════════════╝")
        print("\n".join(configuration_details))

    def parse(self, display_settings=True):
        """Parse and return configuration settings"""
        config = self.collect_arguments()
        config.isTrain = True  # Training mode flag
        config.isVal = False  # Validation mode flag

        if display_settings:
            self.display_configuration(config)

        self.config = config
        return self.config
