import os
import torch
import datetime
import numpy as np

# Import modules with alternative names
from util import set_random_seed as seed_generator
from util import poly_lr as learning_rate_adjuster
from loader import get_val_loader as acquire_validation_dataset
from config import ConfigurationManager as Configurator
from model import model as DeepLearningModel

# Configure image loading behavior
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_validation_settings():
    """Create specialized configuration for model assessment"""
    settings = Configurator().parse()
    settings.isTrain = False
    settings.isVal = True
    return settings


def assess_model_performance(
        validation_datasets,
        neural_network,
        results_directory
):
    """Evaluate neural network performance across validation datasets"""
    neural_network.eval()
    aggregate_correct = aggregate_samples = 0

    with torch.no_grad():
        for dataset in validation_datasets:
            ai_correct = natural_correct = 0

            dataset_identifier = dataset['name']
            ai_data_loader = dataset['val_ai_loader']
            ai_count = dataset['ai_size']
            natural_data_loader = dataset['val_nature_loader']
            natural_count = dataset['nature_size']

            print(f"Evaluating dataset: {dataset_identifier}")

            # Analyze AI-generated images
            for image_batch, target_labels in ai_data_loader:
                image_batch = image_batch.cuda()
                target_labels = target_labels.cuda()

                predictions = neural_network(image_batch)
                prediction_scores = torch.sigmoid(predictions).flatten()

                # Determine correct classifications
                correct_predictions = (
                        ((prediction_scores > 0.5) & (target_labels == 1)) |
                        ((prediction_scores < 0.5) & (target_labels == 0))
                )
                ai_correct += correct_predictions.sum().item()

            ai_performance = ai_correct / ai_count
            print(f"AI Classification Accuracy: {ai_performance:.4f}")

            # Analyze natural images
            for image_batch, target_labels in natural_data_loader:
                image_batch = image_batch.cuda()
                target_labels = target_labels.cuda()

                predictions = neural_network(image_batch)
                prediction_scores = torch.sigmoid(predictions).flatten()

                correct_predictions = (
                        ((prediction_scores > 0.5) & (target_labels == 1)) |
                        ((prediction_scores < 0.5) & (target_labels == 0))
                )
                natural_correct += correct_predictions.sum().item()

            natural_performance = natural_correct / natural_count
            print(f"Natural Image Accuracy: {natural_performance:.4f}")

            # Compute dataset-level performance
            dataset_performance = (ai_correct + natural_correct) / (ai_count + natural_count)
            aggregate_correct += ai_correct + natural_correct
            aggregate_samples += ai_count + natural_count

            print(f"Dataset: {dataset_identifier}, Performance: {dataset_performance:.4f}")

    # Compute overall performance
    overall_performance = aggregate_correct / aggregate_samples
    print(f"Global Accuracy: {overall_performance:.4f}")


def configure_computation_device(device_id):
    """Set computational hardware environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"Selected computation device: GPU {device_id}")


def execute_evaluation_procedure():
    """Main evaluation workflow execution"""
    # Initialize random number generation
    seed_generator()

    # Load configuration settings
    primary_config = Configurator().parse()
    validation_config = generate_validation_settings()

    # Prepare validation data
    print('Preparing validation datasets...')
    validation_datasets = acquire_validation_dataset(validation_config)

    # Configure hardware environment
    configure_computation_device(primary_config.gpu_id)

    # Initialize neural architecture
    network_instance = DeepLearningModel().cuda()

    # Load pre-trained parameters if specified
    if primary_config.load is not None:
        network_instance.load_state_dict(torch.load(primary_config.load))
        print(f'Loaded model parameters from {primary_config.load}')

    # Create results storage location
    results_path = primary_config.save_path
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    print("Commencing model evaluation")
    assess_model_performance(validation_datasets, network_instance, results_path)


if __name__ == '__main__':
    execute_evaluation_procedure()