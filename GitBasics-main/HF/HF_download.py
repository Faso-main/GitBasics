import os, logging
from huggingface_hub import HfApi, notebook_login, hf_hub_download


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_from_huggingface(repo_id: str, filename: str, local_dir: str = None) -> str:
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        logging.info(f"Файл '{filename}' успешно загружен из Hugging Face в: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logging.error(f"Ошибка при загрузке '{filename}' из '{repo_id}' с Hugging Face : {e}")
        return None