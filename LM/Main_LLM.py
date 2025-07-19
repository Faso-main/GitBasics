import logging, torch, os
import torch.nn as nn
from time import time
from datetime import datetime, timedelta
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from LM.Google_sheets import gs, on_hold, sorted_sheet
from Main_itr1 import Train_new_model,Developer_mode


MODEL_MASK='DeepPavlov/rubert-base-cased'
MODEL_PATH=os.path.join('LM','models','model.pth') # путь до обученной модели, если такая есть
SERVER_TIME = 0 # переменная работы с часовым поясом(4 по гринвичу при запуске на VPS) 
CONFIDENCE_THRESHOLD=0.76 # граница приемлимой уверенности модели
TRAIN_EPOCHS = 5 # количество эпох при обучении
LEARNING_RATE = 1e-5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# базовое логирование в системе
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Inquiry:
    "класс хранения передаваемого в систему словаря"
    def __init__(self, hashmap: dict):
        self.hashmap = hashmap  # Инициализация словаря вопросов и ответов

    def get_questions(self):
        "метод получения вопросов"
        return list(self.hashmap.keys())

    def get_answer(self, question):
        "метод получения ответов"
        return self.hashmap.get(question)


class BertEmbedding:
    "Embedding - векторизация текста, параметризация модели"
    def __init__(self, model_name=MODEL_MASK):
        # инициализация токенизатора и модели BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # обязательная оценка настраиваемой модели
        self.device = DEVICE
        self.model.to(self.device)  # Перемещаем модель на устройство
        self.cache = {}  # временная структура хранения векторов, кеш системы

    def encode(self, text: str):
        "векторизует входных даннных и возврат его представление в виде вектора"
        if text in self.cache:
            return self.cache[text]  # получение кэшированного вектора
        
        # токенизируем текст и преобразуем в тензор
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():  # выключение градиентов для скорости
            outputs = self.model(**inputs)  # получение выходных данных модели
        
        # вычисляем среднее значение скрытого состояния
        embedding = outputs.last_hidden_state.mean(dim=1)
        'mean(dim=1): Эта операция вычисляет среднее значение всех токенов (по оси 1) для каждой последовательности в пакете'
        self.cache[text] = embedding  # кэширование векторов
        'Здесь предполагается, что у объекта (возможно, класса) есть атрибут cache, который представляет собой словарь или другую структуру данных для хранения ранее вычисленных эмбеддингов.'
        'Эмбеддинг, вычисленный для входного текста (text), сохраняется в кэше. Это удобно, если тот же текст обрабатывается многократно, чтобы избежать повторных вычислений. Таким образом, если вы запросите эмбеддинг для этого же текста снова, вы сможете просто извлечь его из кэша, не выполняя перерасчет.'
        return embedding


class Search:
    "класс поиска ответов по векторному сходству"
    def __init__(self, hashmap: Inquiry, confidence: float, track_time: bool, 
                 show_accuracy: bool, train_model: bool = False, questions=None, answers=None):
        self.hashmap = hashmap  # ссылка на объект Inquiry, как временная структура данных
        self.confidence = confidence  # п уверенности  ответа
        self.track_time = track_time  # флаг отслеживания времени обработки
        self.show_accuracy = show_accuracy  # флаг отображения точности
        self.bert_encoder = BertEmbedding()  # инициализация BERT векторизатора
        
        # если модель требует переобучения
        if train_model and questions is not None and answers is not None:
            self.trainer = BertTrainer(num_epochs=TRAIN_EPOCHS)  # инициализация тренера(кол-во эпоха)
            self.trainer.train(questions, answers, save_path=MODEL_PATH)  # Обучение модели
            self.trainer.load_model(MODEL_PATH)  # Загрузка обученной модели
        else:
            # если модель уже обученная, просто загружаем её
            self.trainer = BertTrainer(num_epochs=TRAIN_EPOCHS)
            self.trainer.load_model(MODEL_PATH)
        
        # векторизация вопросов и сохранение их в словаре
        self.question_embeddings = {question: self.bert_encoder.encode(question) for question in self.hashmap.get_questions()}

    def find_answer(self, user_question: str):
        "поиск ближайшего ответа на вопрос пользователя"
        if self.track_time: start_time = time()  # запоминаем стартовое время
        
        user_embedding = self.bert_encoder.encode(user_question)  # векторизация вопроса пользователя
        best_match_index = -1 # начальные значения для поиска
        best_similarity = -1  # начальные значения для поиска
    
        # ищем вопрос, наиболее схожий с вопросом пользователя
        for index, (question, question_embedding) in enumerate(self.question_embeddings.items()):
            similarity = torch.cosine_similarity(user_embedding, question_embedding).item()  # вычисляем сходство
            if similarity > best_similarity:
                best_similarity = similarity  # обновляем лучшее сходство
                best_match_index = index  # запоминаем индекс лучшего совпадения

        # формирование вывода времени обработки запроса
        time_info = f'\nВремя обработки: {time() - start_time:.2f} секунд' if self.track_time else ''
        
        # проверка коэффициента сходства заданному уровню
        if best_similarity > self.confidence:
            matched_question = self.hashmap.get_questions()[best_match_index] 
            response = self.hashmap.get_answer(matched_question) 
            accuracy_info = f'\nТочность: {best_similarity:.2f}' if self.show_accuracy else ''
            return f'{response}{time_info}{accuracy_info}'
        else:
            # обработки ошибок при низком уровне совпадения
            if user_question[0] != "=":
                gs.pass_data([str(datetime.now() + timedelta(hours=SERVER_TIME)), user_question])  # вывод данных
                response_generic='Извините, я не могу ответить на ваш вопрос, но в скором времени здесь появится номер колл-центра, куда вы можете обратиться.'
                return f'{response_generic}{time_info}'
            #else:
                return f'{response_generic}{time_info}'


class BertTrainer:
    "класс тренировки модели"
    def __init__(self, model_name='DeepPavlov/rubert-base-cased', num_epochs=1, learning_rate=LEARNING_RATE):
        # инициализация токенизатора и модели
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.train()  # режим обучения
        self.device = DEVICE  # процессор: CPU, видеопамять: "cuda" для GPU
        self.model.to(self.device)  # перемещение модели на устройство
        self.num_epochs = num_epochs  # количество эпох для обучения
        self.learning_rate = learning_rate  # уровень обучения
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # оптимизатор(разные алгоритмы оптимизации)
        self.criterion = nn.CosineSimilarity(dim=-1)  # критерий потерь для вычисления сходства

    def train(self, questions, answers, save_path=MODEL_PATH):
        "Обучает модель на предоставленных вопросах и ответах."
        for epoch in range(self.num_epochs):
            total_loss = 0  # общая потеря для текущей эпохи
            for question in questions:
                # векторизация вопроса
                question_embedding = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
                question_embedding = self.model(**question_embedding).last_hidden_state.mean(dim=1)

                # векторизация ответа
                answer_embedding = self.tokenizer(answers[question], return_tensors="pt", padding=True, truncation=True).to(self.device)
                answer_embedding = self.model(**answer_embedding).last_hidden_state.mean(dim=1)

                # вычисление потерь
                loss = 1 - self.criterion(question_embedding, answer_embedding).mean()
                total_loss += loss.item()  # суммирование потерь

                
                self.optimizer.zero_grad() # обнуление градиента
                'В PyTorch градиенты накапливаются по умолчанию. Поэтому, если вы не очищаете их, при каждом вызове loss.backward() градиенты будут накапливаться и добавляться к предыдущим. Это может привести к неправильному обновлению весов'
                loss.backward() #  вычисление градиента функции потерь по отношению к параметрам модели
                self.optimizer.step()  # обновляет параметры модели на основе рассчитанных градиентов и выбранного алгоритма оптимизации

            # логируем счетчик эпох
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {total_loss:.4f}')

        # сохранение модели после обучения
        self.save_model(save_path)

    def save_model(self, path):
        "сохранение модели по указанному пути"
        logging.info(f'Saving model to {path}')
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        "получение модели из указанного пути"
        logging.info(f'Loading model from {path}')
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()  # обязательная оценка настраиваемой модели


# инициализация данных
inquiry = Inquiry(sorted_sheet) 
questions = inquiry.get_questions()  
answers = inquiry.hashmap  #

# флаги настройки тренировочного режима и режима разработчика

# Поиск ответов, на основе входных данных
response = Search(
    inquiry,
    confidence=CONFIDENCE_THRESHOLD,
    track_time=Developer_mode,
    show_accuracy=Developer_mode,
    train_model=Train_new_model,
    questions=questions if Train_new_model else None,
    answers=answers if Train_new_model else None
)
