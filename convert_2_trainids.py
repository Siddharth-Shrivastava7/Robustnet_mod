import os 
from PIL import Image 
import numpy as np
from tqdm import tqdm
import datasets.cityscapes_labels as cityscapes_labels
id_to_trainid = cityscapes_labels.label2trainid

src_path = '/home/sidd_s/scratch/results/robustnet/saved_models/val/pred'
src_files = []
dest_path = src_path.replace('pred', 'pred_trainids')
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

for file in tqdm(os.listdir(src_path)):
    src_file_path =  os.path.join(src_path, file)
    prediction = np.array(Image.open(src_file_path), dtype=np.uint8) 
    label_out = np.zeros_like(prediction)
    for label_id, train_id in cityscapes_labels.label2trainid.items():
        label_out[np.where(prediction == label_id)] = train_id
    label_out = Image.fromarray(label_out)
    label_out.save(os.path.join(dest_path, file), "PNG")