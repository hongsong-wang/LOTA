import os
import torch
import numpy
from datetime import datetime
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import original modules with aliases
import util as toolkit
from loader import get_loader as fetch_train_data, get_val_loader as fetch_val_data
from config import ConfigurationManager as Configurator
from model import model as NeuralNetwork
from util import bceLoss as compute_binary_loss


def prepare_validation_config():
    """Create validation-specific configuration"""
    val_cfg = Configurator().parse()
    val_cfg.isTrain = False
    val_cfg.isVal = True

    return val_cfg


def execute_training_iteration(
        data_provider,
        network,
        optimizer,
        epoch_index,
        storage_location
):
    """Perform training iteration"""
    network.train()
    global iteration_counter
    epoch_iterations = 0
    total_loss = 0

    try:
        for batch_idx, (inputs, targets) in enumerate(data_provider, start=1):
            optimizer.zero_grad()

            # Move data to GPU
            inputs = inputs.cuda()
            targets = targets.cuda()

            # Forward pass
            outputs = network(inputs).ravel()

            # Compute loss
            loss_function = compute_binary_loss()
            batch_loss = loss_function(outputs, targets)

            # Backward pass
            batch_loss.backward()
            optimizer.step()

            # Update counters
            iteration_counter += 1
            epoch_iterations += 1
            total_loss += batch_loss.item()

            # Log progress
            if batch_idx % 500 == 0 or batch_idx == total_batches or batch_idx == 1:
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                progress_percent = (batch_idx / total_batches) * 100

                status_report = (
                    f"📊 Epoch: {epoch_index:02d}/{config.epoch:02d} | "
                    f"🔢 Iteration: {batch_idx:04d}/{total_batches:04d} "
                    f"({progress_percent:.1f}%) | "
                    f"📉 Loss Metric: {batch_loss.item():.6f}"
                )
                print(status_report)

        # Save periodic checkpoint
        if epoch_index % 50 == 0:
            checkpoint_path = os.path.join(
                storage_location,
                f'Network_epoch_{epoch_index}.pth'
            )
            torch.save(network.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        print("Training interrupted: saving model and exiting")


def perform_validation(
        validation_sets,
        network,
        epoch_index,
        storage_location
):
    """Evaluate model on validation sets"""
    network.eval()
    global best_performing_epoch, highest_accuracy

    total_correct = total_samples = 0

    with torch.no_grad():
        for dataset in validation_sets:
            correct_ai = correct_nature = 0

            name = dataset['name']
            ai_loader = dataset['val_ai_loader']
            ai_count = dataset['ai_size']
            nature_loader = dataset['val_nature_loader']
            nature_count = dataset['nature_size']

            print(f"||Validating||")

            # Process AI-generated images
            for inputs, targets in ai_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()

                predictions = network(inputs)
                probabilities = torch.sigmoid(predictions).ravel()

                # Count correct predictions
                correct = (
                        ((probabilities > 0.5) & (targets == 1)) |
                        ((probabilities < 0.5) & (targets == 0))
                )
                correct_ai += correct.sum().item()

            ai_accuracy = correct_ai / ai_count
            #print(f"AI Accuracy: {ai_accuracy:.4f}")

            # Process natural images
            for inputs, targets in nature_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()

                predictions = network(inputs)
                probabilities = torch.sigmoid(predictions).ravel()

                correct = (
                        ((probabilities > 0.5) & (targets == 1)) |
                        ((probabilities < 0.5) & (targets == 0))
                )
                correct_nature += correct.sum().item()

            nature_accuracy = correct_nature / nature_count
            #print(f"Nature Accuracy: {nature_accuracy:.4f}")

            # Calculate dataset accuracy
            dataset_accuracy = (correct_ai + correct_nature) / (ai_count + nature_count)
            total_correct += correct_ai + correct_nature
            total_samples += ai_count + nature_count

            print(f"Epoch: {epoch_index}, Accuracy: {dataset_accuracy:.4f}")

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_samples

    # Save best model
    if epoch_index == 1:
        best_performing_epoch = 1
        highest_accuracy = overall_accuracy
        best_model_path = os.path.join(storage_location, 'Network_best.pth')
        torch.save(network.state_dict(), best_model_path)
        print(f"Saved best model on Epoch: {epoch_index}")
    else:
        if overall_accuracy > highest_accuracy:
            best_performing_epoch = epoch_index
            highest_accuracy = overall_accuracy
            best_model_path = os.path.join(storage_location, 'Network_best.pth')
            torch.save(network.state_dict(), best_model_path)
            print(f"Saved best model on Epoch: {epoch_index}")

    print(
        f"🏆 Performance Report | "
        f"Current Epoch: {epoch_index:03d} | "
        f"Accuracy Score: {overall_accuracy:.2%} | "
        f"Peak Performance: Epoch {best_performing_epoch:03d} | "
        f"Highest Accuracy: {highest_accuracy:.2%}"
    )


def configure_gpu(gpu_id):
    """Set GPU configuration"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def main_execution():
    """Main training procedure"""
    # Initialize environment
    torch.set_num_threads(2)
    toolkit.set_random_seed()

    # Load configurations
    global config
    config = Configurator().parse()
    val_config = prepare_validation_config()

    # Prepare data
    global total_batches
    train_loader = fetch_train_data(config)
    total_batches = len(train_loader)
    val_loader = fetch_val_data(val_config)

    # Configure GPU
    configure_gpu(config.gpu_id)

    # Initialize model
    model = NeuralNetwork().cuda()
    if config.load:
        model.load_state_dict(torch.load(config.load))
        print(f"Loaded model from {config.load}")

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    # Create output directory
    output_dir = config.save_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize training state
    global iteration_counter, best_performing_epoch, highest_accuracy
    iteration_counter = 0
    best_performing_epoch = 0
    highest_accuracy = 0

    print("||Training||")

    # Training loop
    for epoch in range(1, config.epoch + 1):
        # Adjust learning rate
        current_lr = toolkit.poly_lr(optimizer, config.lr, epoch, config.epoch)

        # Training iteration
        execute_training_iteration(
            train_loader, model, optimizer, epoch, output_dir
        )

        # Validation
        perform_validation(
            val_loader, model, epoch, output_dir
        )


if __name__ == '__main__':
    main_execution()
