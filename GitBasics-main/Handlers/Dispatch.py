from aiogram import types, Router
from aiogram.filters.command import Command
from Handlers.Keyboards import create_keyboard


router_Dispatch=Router()

REPLY_ON_DISPATCH=f"Привет! Я бот для помощи абитуриентам и студентам САМГТУ, ты можешь обратиться ко мне с любым вопросом,и я очень постараюсь ответить тебе на него \U0001F49C \U0001F4AB"


@router_Dispatch.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        text=REPLY_ON_DISPATCH,
        reply_markup=create_keyboard())
    
