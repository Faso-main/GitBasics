import csv
import os
import logging
import psutil
from sklearn.metrics import f1_score
import numpy as np

results = {}

Developer_mode = False
Train_new_model = True 
Load_from_huggingface_model = True
SERVER_TIME = 0

def save_results(epoch_num, train_loss, val_accuracy, f1, memory_usage, base_dir):
    global results
    results[epoch_num] = {
        'epoch': epoch_num,
        'train_loss': train_loss,
        'val_accuracy': val_accuracy,
        'f1_score': f1,
        'memory_usage_mb': memory_usage
    }
    csv_file = os.path.join(base_dir, 'model_metrics.csv')
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[epoch_num].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results[epoch_num])
    logging.info(f"Postprocessing results saved for Epoch {epoch_num}: {results[epoch_num]}")