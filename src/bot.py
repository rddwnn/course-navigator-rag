from __future__ import annotations
import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from .config import settings
from .router import router


async def main() -> None:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    dp.include_router(router)

    print("Бот запущен, начинаем polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())