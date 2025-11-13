from __future__ import annotations

from typing import Dict

from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.fsm.context import FSMContext

from .bot_state import ChatStates
from .rag_pipeline import answer_question


router = Router()

AVAILABLE_COURSES: Dict[str, str] = {
    "os-2023": "Operating Systems 2023",
}

def build_courses_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                text=title,
                callback_data=f"course:{course_id}",
            )
        ]
        for course_id, title in AVAILABLE_COURSES.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:

    await state.clear()

    kb = build_courses_keyboard()
    text = (
        "Hello! I am a course assistant bot.\n\n"
        "1. Choose a course below.\n"
        "2. Then ask questions about the materials of this course.\n"
    )

    await message.answer(text, reply_markup=kb)
    await state.set_state(ChatStates.choosing_course)


@router.callback_query(F.data.startswith("course:"))
async def on_course_chosen(callback: CallbackQuery, state: FSMContext) -> None:
    data = callback.data or ""
    _, course_id = data.split(":", maxsplit=1)

    if course_id not in AVAILABLE_COURSES:
        await callback.answer("Unknown course.", show_alert=True)
        return

    await state.update_data(course_id=course_id)
    await state.set_state(ChatStates.chatting)

    await callback.message.edit_reply_markup(reply_markup=None) 
    await callback.message.answer(
        f"Course selected: {AVAILABLE_COURSES[course_id]}\n\n"
        "Now just write a question about the materials of this course."
    )
    await callback.answer()

@router.message(ChatStates.chatting, F.text)
async def handle_question(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    course_id = data.get("course_id")

    if not course_id:
        await message.answer(
            "I don't see the selected course. Please press /start and choose a course again."
        )
        await state.set_state(ChatStates.choosing_course)
        return

    question = message.text.strip()
    if not question:
        await message.answer("Please write a text question.")
        return

    await message.answer("Thinking about the answer...")

    try:
        answer = answer_question(course_id, question)
    except Exception as e: 
        await message.answer(
            "An error occurred while trying to answer your question.\n"
            "Please try again or ask a different question."
        )
        return

    await message.answer(answer)
