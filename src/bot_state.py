from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class ChatStates(StatesGroup):
    choosing_course = State()
    chatting = State()
