"""API-facing training and validation orchestration for ASGAE."""

from pathlib import Path
import torch
import numpy as np
from app.ASGAE import AE


def run_training(params):
    """Train ASGAE using an API parameter object.

    Args:
        params: Object exposing the fields defined by ``app.main.Params``.

    Returns:
        A JSON-serializable completion status.

    Raises:
        ValueError: If ``optimizer`` is neither ``adam`` nor ``sgd``.
    """
    # Configure deterministic random seeds.
    seed = 472
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Select a CUDA device when one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover serialized mesh graphs.
    datadir = params.db_path
    files = [str(x) for x in Path(datadir).glob('*.pt')]

    # Build the dataloader configuration.
    trainDict = {
        'batch_size': max(2,params.batch_size),
        'test_batch_size': max(2, params.test_batch_size),
        'shuffle': True,
        'setSplit': [params.train_set_percentage, params.val_set_percentage, params.test_set_percentage]
    }

    # Each documented graph provides RGB texture, giving 3 position + 3 color channels.
    net = AE(6, params.embedding_dim, k_order=params.k_order, bias=True, Dataset=files, trainDic=trainDict)
    net.to(device)

    # Construct the requested optimizer.
    if params.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    elif params.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=params.learning_rate)
    else:
        raise ValueError(f"Optimizer not supported: {params.optimizer}")

    # Train with validation-based early stopping.
    loss_prev = float('inf')
    counter = 0
    for epoch in range(params.epochs):
        print(f"Epoch {epoch+1}/{params.epochs}")
        loss_train=net.Train_one_epoch(optimizer, device, verbose=params.verbose)
        loss_val=net.Test_one_epoch(device, verbose=params.verbose)
        print(f"Training Loss: {loss_train:.6f} || Validation Loss: {loss_val:.6f}")
        # Early stopping condition (optional)
        if np.abs(loss_val-loss_prev) < 1e-5:
            counter += 1
        else:
            counter = 0
        if counter >= 5:
            print("Early stopping criterion met... \nStopping training.")
            break
        
        if loss_prev >= loss_val:
            torch.save(net.state_dict(), params.model_path)
        loss_prev = loss_val
        

    return {"status": "Training done!", "\tEpochs": params.epochs}

def Run_Test(params):
    """Load a trained ASGAE model and evaluate it on the requested split.

    Args:
        params: Object exposing the fields defined by ``app.main.Params``.

    Returns:
        A JSON-serializable completion status.
    """
    # Configure deterministic random seeds.
    seed = 472
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Select a CUDA device when one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover serialized mesh graphs.
    datadir = params.db_path
    files = [str(x) for x in Path(datadir).glob('*.pt')]

    # Build the evaluation dataloader configuration.
    print("Preparing testing...")
    if params.full_test_dataset:
        print("Using full dataset for testing.")
        datasplit = [0.0, 0.0, 1.0]
    else:
        print("Using a subset of the dataset for testing.")
        datasplit = [params.train_set_percentage, params.val_set_percentage, params.test_set_percentage]
    testDict = {
        'batch_size': max(2, params.test_batch_size),
        'shuffle': False,
        'setSplit': datasplit
    }

    # Each documented graph provides RGB texture, giving 3 position + 3 color channels.
    net = AE(6, params.embedding_dim, k_order=params.k_order, bias=True, Dataset=files, trainDic=testDict)
    net.to(device)

    # Load model weights onto the device selected above.
    net.load_state_dict(torch.load(params.model_path, map_location=device, weights_only=True))

    # Testing
    net.Test_one_epoch(device, verbose=params.verbose, data_loader=net.test_loader)

    return {"status": "Testing done!"}

class Params:
        """Runnable local example configuration matching the FastAPI schema."""
        db_path = '/data'  # directory containing the .pt mesh files
        batch_size = 8
        test_batch_size = 2
        k_order = 4 
        embedding_dim = 512
        optimizer = 'adam'
        learning_rate = 0.001
        epochs = 10
        model_path = './model/ASGAE_model.pt'
        train_set_percentage = 0.8
        val_set_percentage = 0.1
        test_set_percentage = 0.1
        full_test_dataset = False
        verbose = False
        
if __name__ == "__main__":

    params = Params()
    run_training(params)
