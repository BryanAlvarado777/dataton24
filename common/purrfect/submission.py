import os
import glob
from tqdm import tqdm
import torch
import numpy as np
from purrfect.paths import DATASET_PATH
from torch.utils.data import DataLoader
from purrfect.dataset import TestDataset

def create_submission(
    model, submission_name, submission_path="submissions", device="cuda",source="test_public"
):
    model.eval()
    # crea carpeta submission/submission_name/purrfectpredict
    path = os.path.join(submission_path, submission_name, "purrfectpredict")
    os.makedirs(path, exist_ok=True)
    # crea un dataloader con los
    inputs = glob.glob(os.path.join(DATASET_PATH, f"{source}/*.npy"))
    loop = tqdm(inputs, desc=f"Create submission total its {len(inputs)}", leave=False)
    # evala el modelo y crea un archivo .npy por cada prediccion del modelo, el nombre del archivo .npy debe ser el mismo que del archivo de input
    with torch.no_grad():
        for input in loop:
            input_tensor = (
                torch.from_numpy(np.load(input)).to(device).unsqueeze(0).float()
            )
            output = model(input_tensor)
            np.save(
                f"{path}/{input.split('/')[-1]}",
                output.squeeze().detach().cpu().numpy().astype(np.float16),
            )
    # comprimir usando 7zip la carpeta submission/{submission_name}/purrfectpredict con zip, creando un archivo purrfectpredict_submission.zip en la carpeta submission/{submission_name}


from concurrent.futures import ThreadPoolExecutor

def save_prediction(name, output, path):
    """Helper function to save prediction to disk."""
    np.save(os.path.join(path, name), output.astype(np.float16))

def create_submission_v2(
    model, submission_name, submission_path="submissions", device="cuda", batch_size=16
):
    model.eval()
    # Create submission directory
    path = os.path.join(submission_path, submission_name, "purrfectpredict")
    os.makedirs(path, exist_ok=True)

    test_loader = DataLoader(TestDataset(), batch_size=batch_size, shuffle=False)
    loop = tqdm(
        test_loader, desc=f"Create submission total its {len(test_loader)}", leave=True
    )
    
    # Use ThreadPoolExecutor for parallel disk writes
    with torch.no_grad(), ThreadPoolExecutor() as executor:
        futures = []
        for names, batch in loop:
            output = model(batch.to(device))
            for index, name in enumerate(names):
                # Submit save task to executor
                futures.append(executor.submit(save_prediction, name, output[index].squeeze().detach().cpu().numpy(), path))
        
        # Wait for all futures to complete (optional)
        for future in futures:
            future.result()  # This will raise exceptions if any occurred

