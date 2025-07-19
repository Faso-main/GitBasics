from aiogram import types, Router, F
from aiogram.fsm.context import FSMContext
from Handlers.Keyboards import create_keyboard, make_row_keyboard
from LM.LLM_TEST import response
from datetime import datetime, timedelta
from aiogram.types import ReplyKeyboardRemove
from aiogram.fsm.state import State,StatesGroup
from LM.Google_sheets import gs
from Main_itr1 import SERVER_TIME
import logging, re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Survey(StatesGroup):
    question = State()
    direct_to_LLM=State()

router_Survey = Router()

response_generic = 'Извините, я не могу ответить на ваш вопрос, но в скором времени здесь появится номер колл-центра, куда вы можете обратиться.'

spravka = [
    "Справка, что студент",
    "Справка-вызов",
    "Справка об обучении (периоде)",
    "Справка о размере стипендии",
    "Справка в военкомат",
]

grants = [
    "размер стипендии",
    "социальная стипендия",
    "информация о стипендии",
]

dekanat=[
    "часы работы деканата",
    "время работы деканата",
    "связь с деканатом",
    "телефон деканата"
]

combines_prompt_array=spravka + grants + dekanat


@router_Survey.callback_query(F.data == "table")
async def cmd_question(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.answer(
        text="Возможность просмотра расписания появится в боте позднее\nЖдем и верим в моих разработчиков!💻")
    await callback.answer()


@router_Survey.callback_query(F.data == "wi-fi")
async def cmd_wifi(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.answer(
        text="Сеть: samgtu_emp пароль: ooChaThee4\n"
        "Сеть: samgtu_student пароль: cueX2CBH\n"
        "Сеть: samgtu_guest пароль:see8Zu6k")
    await callback.answer()


@router_Survey.callback_query(F.data == "question")
async def cmd_question(callback: types.CallbackQuery, state: FSMContext):
    await state.set_state(Survey.question)
    await callback.message.answer(text="Какой у тебя вопрос?")


@router_Survey.message(Survey.question)
async def handle_question(message: types.Message, state: FSMContext):
    try:
        user_message_text=message.text
        if not user_message_text or not user_message_text.strip():
            await message.answer("Пожалуйста, напишите ваш вопрос.")
            return

        if user_message_text.startswith('/'):
            await message.answer("Пожалуйста, используйте текстовые сообщения для вопросов. Команды бота доступны через меню.")
            return
        
        user_message = user_message_text.lower()

        if len(user_message) > 100:
            await message.answer("Пожалуйста, напишите сообщение короче 100 символов.")
            await state.set_state(Survey.question)
            return

        if re.search(r"[<>{}$#@&*!%\\/|^`~;\]\[\(\)]|\s{2,}|[\U0001F300-\U0001F6FF\U0001F1E0-\U0001F1FF]", user_message):
            await message.answer(text="Пожалуйста, используйте только текст без специальных символов.")
            await state.set_state(Survey.question)
            return

        if "справка" in user_message:
            await message.answer("Уточни, пожалуйста, какая справка:", reply_markup=make_row_keyboard(spravka))
            await state.set_state(Survey.direct_to_LLM)
        elif "стипендия" in user_message:
            await message.answer("Уточни, пожалуйста, какая информация о стипендии тебя интересует:",
                                 reply_markup=make_row_keyboard(grants))
            await state.set_state(Survey.direct_to_LLM)
        elif "деканат" in user_message:
            await message.answer("Уточни, пожалуйста, какая информация о деканате тебя интересует:",
                                 reply_markup=make_row_keyboard(dekanat))
            await state.set_state(Survey.direct_to_LLM)
        else:
            await process_general_question(message, state)

    except Exception as e:
        logger.error(f"Unhandled error in handle_question for message '{message.text}': {e}", exc_info=True) # Log traceback
        await message.answer("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")


@router_Survey.message(Survey.direct_to_LLM)
async def process_general_question(message: types.Message, state: FSMContext):
    await message.answer(text="Ваш запрос обрабатывается")
    try:
        answer = response.find_answer(message.text)
        if not answer:
            # LM did not find an answer
            # Log the unanswered query to Google Sheets
            gs.pass_data([str(datetime.now() + timedelta(hours=SERVER_TIME)), str(message.text)])  # вывод данных
            await message.answer(text=response_generic, reply_markup=create_keyboard())
        else:
            # LM found a an answer
            await message.answer(text=answer, reply_markup=ReplyKeyboardRemove())
            await message.answer(text="У тебя остались вопросы?", reply_markup=create_keyboard())
    except Exception as e:
        logger.error(f"Unhandled error in process_general_question for message '{message.text}': {e}", exc_info=True) # Log traceback
        await message.answer(text="Произошла ошибка при обработке запроса. Попробуйте позже.")



@router_Survey.message(Survey.direct_to_LLM, F.text.in_(combines_prompt_array))
async def handle_spravka_grants_decanat(message: types.Message, state: FSMContext):
    try:
        answer = response.find_answer(message.text.lower())
        if not answer:
            gs.pass_data([str(datetime.now() + timedelta(hours=SERVER_TIME)), str(message.text)])  # вывод данных
            await message.answer(text=response_generic, reply_markup=create_keyboard())
        else:
            await message.answer(answer)
            await message.answer("У тебя остались вопросы?", reply_markup=create_keyboard())
    except Exception as e:
        logger.error(f"Unhandled error in handle_spravka_grants_decanat for message '{message.text}': {e}", exc_info=True) # Log traceback
        await message.answer(text="Произошла ошибка при обработке запроса. Попробуйте позже.")


