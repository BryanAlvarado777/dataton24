import os
import glob
from tqdm import tqdm
import torch
import numpy as np
from purrfect.paths import DATASET_PATH
def create_submission(model, submission_name,submission_path="submissions",device="cuda"):
  model.eval()
  #crea carpeta submission/submission_name/purrfectpredict
  path = os.path.join(submission_path,submission_name,"purrfectpredict")
  os.makedirs(path, exist_ok=True)
  #crea un dataloader con los
  inputs = glob.glob(os.path.join(DATASET_PATH,f"test_public/*.npy"))
  loop = tqdm(inputs, desc=f'Create submission total its {len(inputs)}', leave=False)
  #evala el modelo y crea un archivo .npy por cada prediccion del modelo, el nombre del archivo .npy debe ser el mismo que del archivo de input
  with torch.no_grad():
    for input in loop:
      input_tensor = torch.from_numpy(np.load(input)).to(device).unsqueeze(0).float()
      output = model(input_tensor)
      np.save(f"{path}/{input.split('/')[-1]}", output.squeeze().detach().cpu().numpy().astype(np.float16))
  #comprimir usando 7zip la carpeta submission/{submission_name}/purrfectpredict con zip, creando un archivo purrfectpredict_submission.zip en la carpeta submission/{submission_name}
