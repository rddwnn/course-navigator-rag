import asyncio

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

from .config import settings  # относительный импорт


bot = Bot(token=settings.telegram_bot_token)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я RAG-бот для курсов.\n"
        "Сейчас проект только инициализирован — скоро научусь отвечать по материалам курса."
    )


async def main() -> None:
    # здесь потом добавим router, middlewares и т.д.
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
