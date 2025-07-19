from aiogram import types,Router, F
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardRemove


router_Help=Router()

REPLY_TO_HELP=f'Чтобы задать мне вопрос, нажми на кнопку "Задать вопрос" в меню\nБуду рад помочь!😄'


@router_Help.message(Command("help"))
async def answer_on_help(message: types.Message,state:FSMContext):
    await state.clear()
    await message.answer(text=REPLY_TO_HELP, reply_markup=ReplyKeyboardRemove())

@router_Help.message(F.text)
async def answer_on_help(message: types.Message,state:FSMContext):
    await state.clear()
    await message.answer(text=REPLY_TO_HELP, reply_markup=ReplyKeyboardRemove())

