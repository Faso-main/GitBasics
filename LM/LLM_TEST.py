"""Здесь находится валидация модели"""



import logging, torch, os
import csv
import torch.nn as nn
from time import time
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import psutil
from sklearn.metrics import f1_score
from datetime import datetime, timedelta
import random
import numpy as np

from LM.Google_sheets import gs, on_hold, sorted_sheet
from Main_itr1 import Train_new_model, Developer_mode, Load_from_huggingface_model
from HF.HF_upoad import upload_to_huggingface
from HF.HF_download import download_from_huggingface

base_dir = "LM/Validation"

os.makedirs(base_dir, exist_ok=True)

questions_list = list(sorted(sorted_sheet.keys()))
train_questions, val_questions = train_test_split(questions_list, test_size=0.2, random_state=42)
val_answers = {q: sorted_sheet[q] for q in val_questions}
train_answers = {q: sorted_sheet[q] for q in train_questions}


# Конфигурация системы
class Config:
    MODEL_MASK = 'DeepPavlov/rubert-base-cased'
    MODEL_NAME = f'model_{str(datetime.now().strftime("%Y_%m_%d"))}.pth'
    MODEL_PATH = os.path.join('LM', 'models', MODEL_NAME)
    SERVER_TIME = 0
    CONFIDENCE_THRESHOLD = 0.76
    TRAIN_EPOCHS = 3
    LEARNING_RATE = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    VAL_SPLIT_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_SEQ_LENGTH = 512
    HF_REPO_NAME = f'faso312/SamGTU_Bert_QA_Model'
    HF_MODEL_FILENAME = 'latest_model.pth'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Inquiry:
    "Класс хранения передаваемого в систему словаря вопросов и ответов"
    def __init__(self, hashmap: dict):
        self.hashmap = hashmap

    def get_questions(self):
        return list(self.hashmap.keys())

    def get_answer(self, question):
        return self.hashmap.get(question)


class BertEmbedding:
    "Embedding - векторизация текста, параметризация модели"
    def __init__(self, model_name=Config.MODEL_MASK):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.device = Config.DEVICE
        self.model.to(self.device)
        self.cache = {}

    def encode(self, text: str):
        if text in self.cache:
            return self.cache[text]

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)
        self.cache[text] = embedding
        return embedding


class QADataset(Dataset):
    def __init__(self, questions: list, answers_map: dict):
        self.questions = questions
        self.answers_map = answers_map

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers_map[question]
        return question, answer


class BertTrainer:
    "Класс тренировки модели"
    def __init__(self, model_name=Config.MODEL_MASK, num_epochs=Config.TRAIN_EPOCHS, learning_rate=Config.LEARNING_RATE):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.train()
        self.device = Config.DEVICE
        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CosineSimilarity(dim=-1)

    def train(self, questions_list, answers_map, save_path=Config.MODEL_PATH):
        train_questions, val_questions = train_test_split(
            questions_list, test_size=Config.VAL_SPLIT_SIZE, random_state=Config.RANDOM_STATE
        )

        train_dataset = QADataset(train_questions, answers_map)
        val_dataset = QADataset(val_questions, answers_map)

        train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0

            for batch_questions, batch_answers in train_dataloader:
                question_inputs = self.tokenizer(
                    list(batch_questions), return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH
                ).to(self.device)
                question_embedding = self.model(**question_inputs).last_hidden_state.mean(dim=1)

                answer_inputs = self.tokenizer(
                    list(batch_answers), return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH
                ).to(self.device)
                answer_embedding = self.model(**answer_inputs).last_hidden_state.mean(dim=1)

                loss = 1 - self.criterion(question_embedding, answer_embedding).mean()
                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}')

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_questions, batch_answers in val_dataloader:
                    question_inputs = self.tokenizer(
                        list(batch_questions), return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH
                    ).to(self.device)
                    question_embedding = self.model(**question_inputs).last_hidden_state.mean(dim=1)

                    answer_inputs = self.tokenizer(
                        list(batch_answers), return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH
                    ).to(self.device)
                    answer_embedding = self.model(**answer_inputs).last_hidden_state.mean(dim=1)

                    loss = 1 - self.criterion(question_embedding, answer_embedding).mean()
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Validation Loss: {avg_val_loss:.4f}')


            self.postprocess(val_questions, answers_map, avg_train_loss, epoch + 1)

        self.save_model(save_path)

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024 ** 2

    def postprocess(self, val_questions, val_answers, train_loss, epoch_num):
        self.model.eval()

        predictions = []
        true_labels = []

        all_answers = list(val_answers.values())

        with torch.no_grad():
            for i, question in enumerate(val_questions):
                question_inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(
                    self.device)
                try:
                    question_outputs = self.model(**question_inputs)
                    question_embedding = question_outputs.last_hidden_state.mean(dim=1)
                except Exception as e:
                    logging.error(f"Error processing validation question: {question}, Error: {e}")
                    continue

                correct_answer = val_answers[question]
                answer_inputs = self.tokenizer(correct_answer, return_tensors="pt", padding=True, truncation=True).to(
                    self.device)
                try:
                    answer_outputs = self.model(**answer_inputs)
                    answer_embedding = answer_outputs.last_hidden_state.mean(dim=1)
                except Exception as e:
                    logging.error(f"Error processing validation answer for question: {question}, Error: {e}")
                    continue

                similarity_positive = self.criterion(question_embedding, answer_embedding).item()
                predictions.append(1 if similarity_positive > 0.76 else 0)
                true_labels.append(1)

                filtered_answers = [ans for ans in all_answers if ans != correct_answer]
                if filtered_answers:
                    negative_answer = random.choice(filtered_answers)
                    neg_answer_inputs = self.tokenizer(negative_answer, return_tensors="pt", padding=True,
                                                       truncation=True).to(self.device)
                    try:
                        neg_answer_outputs = self.model(**neg_answer_inputs)
                        neg_answer_embedding = neg_answer_outputs.last_hidden_state.mean(dim=1)
                    except Exception as e:
                        logging.error(f"Error processing negative answer for question: {question}, Error: {e}")
                        continue

                    similarity_negative = self.criterion(question_embedding, neg_answer_embedding).item()
                    predictions.append(1 if similarity_negative > 0.76 else 0)
                    true_labels.append(0)

        val_accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        f1 = f1_score(true_labels, predictions, average='weighted')
        memory_usage = self.get_memory_usage()

        results = {
            'epoch': epoch_num,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'f1_score': f1,
            'memory_usage_mb': memory_usage
        }

        csv_file = os.path.join(base_dir, 'model_metrics.csv')
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

        logging.info(f"Postprocessing results saved for Epoch {epoch_num}: {results}")

    def save_model(self, path):
        logging.info(f'Saving model to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        if hasattr(Config, 'HF_REPO_NAME'):
            upload_to_huggingface(
                Config.MODEL_PATH,
                Config.HF_MODEL_FILENAME,
                Config.HF_REPO_NAME,
                Config.TRAIN_EPOCHS
            )

    def load_model(self, path):
        logging.info(f'Loading model from {path}')
        if not os.path.exists(path):
            logging.error(f"Модель не найдена по пути: {path}. Убедитесь, что модель обучена и сохранена.")
            return False
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logging.info(f"Модель успешно загружена локально из: {path}")
            return True
        except Exception as e:
            logging.error(f"Ошибка при загрузке локальной модели из {path}: {e}")
            return False

    def load_from_huggingface(self, repo_id, filename):
        logging.info(f'Loading model from Hugging Face: {repo_id}/{filename}')
        try:
            model_file_path = download_from_huggingface(repo_id=repo_id, filename=filename)
            if model_file_path:
                self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
                self.model.eval()
                logging.info(f"Модель успешно загружена из Hugging Face: {model_file_path}")
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Не удалось загрузить модель из Hugging Face: {e}")
            return False


class Search:
    "Класс поиска ответов по векторному сходству"
    def __init__(self, hashmap: Inquiry, confidence: float, track_time: bool,
                 show_accuracy: bool, train_model: bool = False,
                 load_from_hf: bool = False,
                 questions=None, answers=None):
        self.hashmap = hashmap
        self.confidence = confidence
        self.track_time = track_time
        self.show_accuracy = show_accuracy
        self.bert_encoder = BertEmbedding()

        self.trainer = BertTrainer()

        if train_model and questions is not None and answers is not None:
            logging.info("Начинается обучение новой модели...")
            self.trainer.train(questions, answers, save_path=Config.MODEL_PATH)
            logging.info("Обучение модели завершено. Загрузка обученной модели.")
        elif load_from_hf:
            logging.info("Попытка загрузить модель из Hugging Face.")
            if not self.trainer.load_from_huggingface(Config.HF_REPO_NAME, Config.HF_MODEL_FILENAME):
                logging.warning("Загрузка из Hugging Face не удалась. Попытка загрузить модель локально.")
                if not self.trainer.load_model(Config.MODEL_PATH):
                    logging.error("Не удалось загрузить модель ни локально, ни из Hugging Face. Система может работать некорректно.")
        else:
            logging.info("Загрузка модели из локального хранилища (по умолчанию).")
            if not self.trainer.load_model(Config.MODEL_PATH):
                logging.error("Не удалось загрузить модель локально. Система может работать некорректно.")

        logging.info("Векторизация вопросов базы знаний...")
        self.question_embeddings = {
            question: self.bert_encoder.encode(question) for question in self.hashmap.get_questions()
        }
        logging.info("Векторизация вопросов завершена.")

    def find_answer(self, user_question: str):
        if self.track_time: start_time = time()

        user_embedding = self.bert_encoder.encode(user_question)
        best_match_index = -1
        best_similarity = -1

        for index, (question, question_embedding) in enumerate(self.question_embeddings.items()):
            similarity = torch.cosine_similarity(user_embedding, question_embedding).item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = index

        time_info = f'\nВремя обработки: {time() - start_time:.2f} секунд' if self.track_time else ''

        if best_similarity > self.confidence:
            matched_question = self.hashmap.get_questions()[best_match_index]
            response = self.hashmap.get_answer(matched_question)
            accuracy_info = f'\nТочность: {best_similarity:.2f}' if self.show_accuracy else ''
            return f'{response}{time_info}{accuracy_info}'
        else:
            if user_question and user_question[0] != "=":
                gs.pass_data([str(datetime.now() + timedelta(hours=Config.SERVER_TIME)), user_question])
            response_generic = 'Извините, я не могу ответить на ваш вопрос, но в скором времени здесь появится номер колл-центра, куда вы можете обратиться.'
            return f'{response_generic}{time_info}'


inquiry = Inquiry(sorted_sheet)
questions_for_training = inquiry.get_questions()
answers_map_for_training = inquiry.hashmap


response = Search(
    inquiry,
    confidence=Config.CONFIDENCE_THRESHOLD,
    track_time=Developer_mode,
    show_accuracy=Developer_mode,
    train_model=Train_new_model,
    load_from_hf=Load_from_huggingface_model,
    questions=questions_for_training if Train_new_model else None,
    answers=answers_map_for_training if Train_new_model else None
)
