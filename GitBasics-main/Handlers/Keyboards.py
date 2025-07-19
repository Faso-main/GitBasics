from aiogram.utils.keyboard import InlineKeyboardBuilder,ReplyKeyboardBuilder
from aiogram import types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

def make_row_keyboard(items: list[str], max_columns: int = 2) -> ReplyKeyboardMarkup:
    # Проверка, что items должен быть списком
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError("Input must be a list of strings.")

    # Создание строк для клавиатуры
    rows = []
    for i in range(0, len(items), max_columns):
        row = [KeyboardButton(text=item) for item in items[i:i + max_columns]]
        rows.append(row)

    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)

def create_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text="Задать вопрос",callback_data="question"))
    builder.add(types.InlineKeyboardButton(text="Расписание",callback_data="table"))
    return builder.as_markup()


def create_keyboard_spravka():
    builder = InlineKeyboardBuilder()

    builder.add(types.InlineKeyboardButton(text="Студент",callback_data="spravka"))
    builder.add(types.InlineKeyboardButtontext(text="Вызов",callback_data="spravka"))
    builder.add(types.InlineKeyboardButton(text="Обучение",callback_data="spravka"))
    builder.add(types.InlineKeyboardButton(text="Стипендия",callback_data="spravka"))
    builder.add(types.InlineKeyboardButton(text="Военкомат",callback_data="spravka"))
    return builder.as_markup()

def create_keyboard_grants():
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text="размер стипендии",callback_data="grant"))
    builder.add(types.InlineKeyboardButton(text="социальная стипендия",callback_data="grant"))
    builder.add(types.InlineKeyboardButton(text="информация о стипендии",callback_data="grant"))
    return builder.as_markup()
