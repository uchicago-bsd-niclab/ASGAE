
from pathlib import Path
import torch
import numpy as np
from app.ASGAE import AE


def run_training(params):
    # Seed Configuration
    seed = 472
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Config divice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find dataset files
    datadir = params.db_path
    files = [str(x) for x in Path(datadir).glob('*.pt')]

    # Config dictionary for training
    trainDict = {
        'batch_size': max(2,params.batch_size),
        'test_batch_size': max(2, params.test_batch_size),
        'shuffle': True,
        'setSplit': [params.train_set_percentage, params.val_set_percentage, params.test_set_percentage]
    }

    # Initialize network
    net = AE(params.k_order, params.embedding_dim, bias=True, Dataset=files, trainDic=trainDict)
    net.to(device)

    # Optimizer configuration
    if params.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    elif params.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=params.learning_rate)
    else:
        raise ValueError(f"Optimizer not supported: {params.optimizer}")

    # Training loop
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
    # Seed Configuration
    seed = 472
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Config divice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find dataset files
    datadir = params.db_path
    files = [str(x) for x in Path(datadir).glob('*.pt')]

    # Config dictionary for testing
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

    # Initialize network
    net = AE(params.k_order, params.embedding_dim, bias=True, Dataset=files, trainDic=testDict)
    net.to(device)

    # Load pre-trained model weights
    net.load_state_dict(torch.load(params.model_path))

    # Testing
    net.Test_one_epoch(device)

    return {"status": "Testing done!"}

class Params:
        db_path = '/path/to/your/data'
        batch_size = 6
        test_batch_size = 2
        k_order = 6
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