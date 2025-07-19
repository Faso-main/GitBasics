from huggingface_hub import HfApi, Repository, notebook_login # pip install huggingface_hub, ipywidgets
import os, json
from datetime import datetime
import pandas as pd
# дополнительно нужно пройти авторизацию по ключу(>> huggingface-cli login), за ключом в личку
# >> git config --global credential.helper store

"""
Добавить стороки ниже в ключевой if __name__ после main()
HF_REPO_NAME = "faso312/test1"  # Прописал свое имя, свой аккаунт, возможно стоит сделать под это аккаунт команды
upload_to_huggingface(SAVE_PATH, HF_REPO_NAME)
"""

# Добавьте эту функцию в ваш код
def upload_to_huggingface(model_path, model_name,repo_name, epochs):
    # Аутентификация
    notebook_login()
    
    # Создаем репозиторий
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    
    # Загружаем модель
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_name,
        repo_id=repo_name,
        repo_type="model"
    )
    
    readme_content = f"""
# {repo_name.split('/')[-1]}

This is a model repository for {repo_name.split('/')[-1]}.

## Model Details
[Телеграм бот СамГТУ]

## Training
This model was trained with [PyTorch, Transformers].
[Количество эпох {epochs}].

## Results
В разработке

## Usage
В разработке

## Files

  - `model_path/`: Contains the model weights and configuration.
"""

    temp_readme_path = os.path.join('Handlers','temp_README.md')
    with open(temp_readme_path, "w", encoding="utf-8") as temp: temp.write(readme_content)
    api.upload_file(
        path_or_fileobj=temp_readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    os.remove(temp_readme_path) # удаляем временный файл
    print(f"All files uploaded to https://huggingface.co/{repo_name}")