import asyncio, os, logging
from aiogram import Bot, Dispatcher
from Handlers import Dispatch, Help, Survey
from Handlers.Encoder import decoded_bot_key, test_key
from multiprocessing import freeze_support
from LM.Res import results



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)

logger = logging.getLogger()

token = os.getenv("TELEGRAM_BOT_TOKEN", test_key)
bot = Bot(token)
dp = Dispatcher()

async def main() -> None:
    try:
        dp.include_router(Survey.router_Survey)
        dp.include_router(Dispatch.router_Dispatch)
        dp.include_router(Help.router_Help)
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
        print(f"Results of validation: {results}")
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        await bot.close()

if __name__ == '__main__':
    print('Digital assistant is running.............')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"Error: {e}")