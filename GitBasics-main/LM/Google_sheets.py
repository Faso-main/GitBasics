import gspread, time, functools, numpy, os
from datetime import datetime, timedelta
from time import time
import sqlalchemy as sa
from datetime import datetime as dt
from sqlalchemy.orm import sessionmaker
from BD.SqlFunctions import SqlFunctions
from BD.Models import Base

key__path=os.path.join("LM","Key.json") 

def on_hold(seconds: int): time.sleep(seconds) #задержка для передачи излишних запросов серверу

def processing(func): #универсальный декоратор обработки ошибок
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try: return func(*args, **kwargs)
        except gspread.exceptions.APIError: on_hold(15)
        except Exception as e: print(f'Ошибка вида: {e}.....') #общая обработа ошибок
    return wrapper


class Gspread():
    @processing
    def __init__(self,key_path: str,table_tag: str) -> None:
        self.sa = gspread.service_account(key_path) #подключение в  json файлу библиотеки
        self.sh = self.sa.open(table_tag) #открытие таблицы с таким-то названием

    @processing
    def get_sheet(self, sh_id: int)->list: return self.sh.get_worksheet(sh_id).get_values()[1:]

    @processing
    def get_sheet_by_title(self, sh_tag: str)->list: return self.sh.worksheet(sh_tag).get_values()[1:]
        
    @processing
    def pass_data(self,user_data: list): #передача данных пользоватея в таблицу
        quetions_sh=self.sh.worksheet("Вопросы без ответов")
        last_row = len(quetions_sh.get_values()) + 1 #получение последнего значения заполненной строки +1 
        for col in range(1, len(user_data)+1): #заполнение (1, длинна списка ответов)
            quetions_sh.update_cell(last_row, col, user_data[col-1]) #определяем место ввода(последняя свободная, столбец, значение)


gs=Gspread(key__path,'База знаний для ЦА')


def sorting_gspread(hashmap):
    try:
        np_data = numpy.array(hashmap)
        result_dict = {row[0]: row[1].replace('\n', ' ') for row in np_data}
        return result_dict
    except TypeError: print(f'Ошибка при обращение к ключу google sheets[Key.json]')

questions_IAIT=gs.get_sheet(0)
sorted_sheet=sorting_gspread(questions_IAIT)



db_directory=os.path.join('BD','BaseData')
#db_directory = 'C:/Users/artem/Desktop/Digital_Assistants-main/BaseData'


try: os.makedirs(db_directory, exist_ok=True)
except Exception as e: print(f"Ошибка при создании директории: {e}")

# Создание URL соединения с базой данных
db_path = os.path.join(db_directory, 'QuestAnswer.db')
engine = sa.create_engine(f"sqlite:///{db_path}")
print(engine)

# Создаем таблицу в базе данных
Base.metadata.create_all(engine)

# Создаем сессию для взаимодействия с базой данных
Session = sessionmaker(bind=engine)
session = Session()

for question, answer in sorted_sheet.items():
    SqlFunctions.add_if_not_exists(question, answer, session)

all_data = SqlFunctions.get_all_data(session)
sorted_sheet = {}
for record in all_data:
    sorted_sheet[record.question] = record.answer
