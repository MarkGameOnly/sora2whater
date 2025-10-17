#!/usr/bin/env python3
"""
Advanced Telegram bot for enhancing videos with subtitles and upscaling.

This bot accepts video messages from users, automatically transcribes the
audio using the ``faster‑whisper`` library, generates ASS subtitles with
a configurable style, improves colour and sharpness with FFmpeg filters and
upscales the output to Full HD or 4K depending on the availability of
Real‑ESRGAN.  Users can customise subtitle options such as orientation,
font family, font size and whether subtitles are enabled at all.  A simple
subscription system enforces a free usage limit and allows the
administrator to activate time‑limited access for paying users.  Extended
analytics are available for the administrator to monitor usage and
payments.

Key features
------------

* Automatically transcribes video audio with ``faster‑whisper`` and
  generates ASS subtitles.  Subtitles are styled according to user
  preferences for font family, size and presence.
* Upscales video to 1920×1080 by default.  If the optional
  ``realesrgan‑ncnn‑vulkan`` binary is installed on the system, the
  video is upscaled to 4K (3840×2160) using Real‑ESRGAN.
* Applies gentle colour correction and sharpening via FFmpeg filters.
* Supports both landscape (16:9) and portrait (9:16) output formats.  The
  bot automatically detects the input orientation when the user chooses
  "Auto" or uses the selected orientation to determine subtitle line
  length and scaling.
* Stores user preferences and usage statistics in a JSON file.  Each
  user can customise orientation, font, font size and whether subtitles
  appear.  Users receive a fixed number of free conversions before
  needing to purchase a subscription.  The administrator has unlimited
  conversions and can activate subscriptions for other users.
* Provides an interactive inline keyboard for users to adjust settings
  and for the administrator to view statistics.
* Offers extended analytics for the administrator: total users,
  conversions, payments and new users over the last day, week, month
  and year.  Payments are counted whenever a subscription is activated.

Important notes
---------------

* On Windows, FFmpeg's ``ass`` filter can misinterpret absolute paths
  containing drive letters.  This bot avoids passing absolute paths to
  the filter by running FFmpeg from within the temporary working
  directory and referencing the subtitle file by name only.  This
  technique circumvents the need for complex escaping【65214271342919†L260-L297】.
* The bot relies on ``pysubs2`` for subtitle creation.  Ensure that
  ``pysubs2`` is installed (see ``requirements.txt``) and that the
  selected font is installed on the host system.  If an uninstalled
  font is requested, FFmpeg will silently fall back to a default font.

"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Load environment variables from a .env file if present.  This allows
# sensitive information (bot token, admin ID, API keys) to be stored
# separately from the source code.  The python‑dotenv package reads
# variables defined in a .env file and populates os.environ.  If the
# package is not available (e.g. in some deployment environments), the
# call will have no effect.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if it's not installed, silently continue.
    pass

from aiogram import Bot, Dispatcher, types
from aiogram.types import (
    InputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.utils import executor
from aiogram.types import BotCommand, BotCommandScopeDefault, BotCommandScopeChat

# Optional dependencies
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore

try:
    import pysubs2
except ImportError:
    pysubs2 = None  # type: ignore

__all__ = [
    "BOT_TOKEN",
    "ADMIN_ID",
    "FREE_LIMIT",
    "SUBSCRIPTION_PRICE_USD",
    "PAYMENT_LINK",
    "DATA_FILE",
    "load_data",
    "save_data",
    "get_user",
    "save_user",
    "is_subscribed",
    "is_blocked",
    "add_usage",
    "add_payment",
    "format_subtitles",
    "process_video",
]


# ───────────────────────────────────────────────────────────────────────────────
# Configuration

# Bot token.  The bot token must be provided via the ``BOT_TOKEN``
# environment variable.  We no longer embed the token directly in
# source code.  If the variable is not set, the bot will refuse to
# start.  Use a .env file or set the variable in your hosting
# environment to provide the token.
BOT_TOKEN: str | None = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN environment variable must be set. Define it in your .env file or hosting environment."
    )

# Administrator's Telegram user ID.  The admin has unlimited usage and
# access to management commands.  This value must be provided via
# ``ADMIN_ID`` environment variable.  It is converted to int.  If
# undefined or invalid, the bot will refuse to start.
ADMIN_ID_ENV = os.getenv("ADMIN_ID")
if not ADMIN_ID_ENV:
    raise RuntimeError(
        "ADMIN_ID environment variable must be set. Define it in your .env file or hosting environment."
    )
try:
    ADMIN_ID: int = int(ADMIN_ID_ENV)
except Exception:
    raise RuntimeError("ADMIN_ID must be an integer")

# Number of free conversions each user receives before a subscription is
# required.  The admin is exempt.
FREE_LIMIT: int = 10

# Price of the monthly subscription in USD.  This value is shown to
# users in the profile and help messages.  Note that the actual
# subscription plans below override this for multi‑month options.  A
# monthly plan costs $2.50; a two‑month plan is $5.00; a three‑month
# plan is $9.99 (discounted); a yearly plan is $120.00.
SUBSCRIPTION_PRICE_USD: float = 2.5

# Payment link for purchasing a subscription.  Users should pay through
# this link and then contact the administrator for activation.  The link
# points to your CryptoBot subscription channel.  Adjust this to your
# current channel invite link if needed.
PAYMENT_LINK: str = "https://t.me/send?start=s-sorasub"

# Channel or chat identifier for verifying active subscriptions via channel
# membership.  When a user pays for a plan through CryptoBot, CryptoBot
# automatically invites them to a private channel.  By checking that
# the user is a member of this channel, the bot can treat them as
# subscribed without requiring webhooks.  Replace the default value
# with your channel's username (e.g. "@sorasubscription") or numeric
# chat ID (e.g. -1001234567890).  You can also set the environment
# variable SUBSCRIPTION_CHANNEL to override this value.
SUBSCRIPTION_CHANNEL: str | int = os.getenv("SUBSCRIPTION_CHANNEL", "@sorasubscription")

# Path to the JSON file where all user data, usage statistics and
# preferences are stored.  This file will be created if it does not
# exist.  Data is structured as described below.
DATA_FILE: Path = Path(__file__).parent / "data.json"

# Path to the log file.  All significant actions (subscription
# activations, blocks, warnings, etc.) are recorded here.  Use the
# /logs and /clearlogs commands to view or erase this file.
LOG_FILE: Path = Path(__file__).parent / "logs.txt"

# FFmpeg filters for colour correction and sharpening.  These filters
# gently reduce noise, increase contrast and saturation and sharpen the
# image.  Feel free to adjust the parameters to taste.
FFMPEG_FILTERS: str = (
    # Slight noise reduction
    "hqdn3d=1.0:1.0:6:6,"  # reduce luminance/chrominance noise
    # Mild colour and contrast enhancement
    "eq=brightness=0.05:contrast=1.15:saturation=1.3,"  # richer colours and contrast
    # Sharpening for extra crispness
    "unsharp=7:7:1.0:7:7:0.0"  # wider radius and stronger effect
)

# URL for the external watermark removal service.  This is used in
# the `/watermark` command to provide step‑by‑step instructions for
# removing Sora/Veo watermarks from videos prior to uploading them to
# the bot.  We provide both the original web URL and a Telegram
# MiniApp URL for convenience.  The MiniApp opens the removal tool
# directly within Telegram.  Users can choose either method; the
# instructions will guide them through the process.
WATERMARK_REMOVAL_URL: str = "https://sorrywatermark.com/"

# Telegram MiniApp URL for the watermark removal tool.  When users
# click the “Убрать водяной знак” button in the menu, the bot sends
# them instructions containing a link to this MiniApp.  Using
# Telegram’s Mini App framework, the tool opens in a web view inside
# Telegram, allowing users to upload their watermarked video and
# download the cleaned version without leaving the app.
WATERMARK_MINIAPP_URL: str = "https://t.me/sorrywatermark_bot/sorrywatermarkcom"

# Subscription plans.  Each entry maps a key (used internally) to a
# dictionary containing the number of days the subscription lasts,
# the number of tokens awarded, and the price in USD.  The number of
# tokens roughly corresponds to a generous allowance for generating
# videos.  Feel free to adjust the token amounts or prices to suit
# your business model.  Users can see these values in the /subscribe
# command.  The admin UI also references this structure when
# activating or extending subscriptions.
SUBSCRIPTION_PLANS: Dict[str, Dict[str, Any]] = {
    # One‑month subscription: 30 days, 1000 tokens, $2.50
    "1m": {"days": 30, "tokens": 1000, "price": 2.50},
    # Two‑month subscription: 60 days, 2000 tokens, $5.00
    "2m": {"days": 60, "tokens": 2000, "price": 5.00},
    # Three‑month subscription: 90 days, 3000 tokens, $9.99 (discounted)
    "3m": {"days": 90, "tokens": 3000, "price": 9.99},
    # One‑year subscription: 365 days, 12000 tokens, $120.00
    "1y": {"days": 365, "tokens": 12000, "price": 120.00},
}

# Number of tokens awarded to a referrer when a new user joins using
# their referral link.  The referrer receives these tokens
# immediately; the new user does not receive any extra tokens
# automatically.  Referrals are recorded per user in the 'partners'
# list on the referrer's record.
REFERRAL_BONUS_TOKENS: int = 100

# Cost in tokens of processing a video at different output
# resolutions.  These costs are subtracted from the user's token
# balance for each completed conversion.  Users can process videos
# using free conversions while they still have free usage quota.  Once
# the free quota is exhausted, tokens will be consumed.  If a user
# requests a resolution but lacks sufficient tokens, the request
# automatically downgrades to 1080p (unless they are subscribed).
TOKENS_PER_QUALITY: Dict[str, int] = {
    "1080": 25,
    "2k": 50,
    "4k": 100,
}

# A set of fonts offered to users.  These names should correspond to
# fonts installed on your system.  The keys are the labels shown in the
# UI; the values are the font names passed to ASS.  You can modify or
# extend this mapping to include other installed fonts.
AVAILABLE_FONTS: Dict[str, str] = {
    "Times New Roman": "Times New Roman",
    "Arial": "Arial",
    "Helvetica": "Helvetica",
    "Courier New": "Courier New",
    "DejaVu Sans": "DejaVu Sans",
}

# Default user preferences.  These values are applied when a new user
# interacts with the bot for the first time.  Orientation 'auto'
# detects the input video's aspect ratio to choose landscape or
# portrait processing; font and size select a default style; subtitles
# are enabled by default.
DEFAULT_PREFS = {
    "orientation": "auto",    # one of 'auto', 'landscape', 'portrait'
    "font": "Times New Roman",  # font label (key from AVAILABLE_FONTS)
    "font_size": 12,            # size in points; small value keeps
                                 # text from filling the screen
    "subtitles": True,          # whether to overlay subtitles
    "quality": "1080",          # output quality: '1080', '2k', or '4k'
    "blocked_until": None,      # timestamp until which the user is blocked
    "subscribed_until": None,   # timestamp until which subscription is valid
    "usage": 0,                 # number of conversions used
    "timestamps": [],           # list of conversion timestamps
    "payments": [],             # list of payment timestamps
    "warned_until": None,        # last subscription expiry for which user was warned

    # Token balance.  Users can accumulate tokens via subscriptions
    # or referrals.  Each conversion consumes a number of tokens
    # defined in TOKENS_PER_QUALITY.  Tokens are awarded when a
    # subscription is activated or when another user registers via
    # their referral link.
    "tokens": 0,
    # Referral ID of the user who invited this user, if any.  Used
    # internally to credit the referrer and prevent duplicate
    # referrals.
    "referrer": None,
    # List of user IDs that this user has invited.  Used to track
    # partners and prevent duplicate counting.  Each partner results
    # in REFERRAL_BONUS_TOKENS being added to the referrer's token
    # balance.
    "partners": [],
}

# ──────────────────────────────────────────────────────────────────────────────
# User interface: persistent reply keyboard

# A reply keyboard for quick access to common actions.  This keyboard
# appears below the message input field and mirrors the style of the
# referenced 'Каллорит' bot.  The buttons trigger handlers defined
# later in this file.
MAIN_REPLY_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🎬 Отправить видео")],
        [KeyboardButton("🛡 Убрать водяной знак")],
        [KeyboardButton("🔧 Настройки"), KeyboardButton("👤 Профиль")],
        [KeyboardButton("ℹ️ Помощь")],
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
)


# ───────────────────────────────────────────────────────────────────────────────
# Data persistence helpers

def load_data() -> Dict[str, Any]:
    """Load persistent data from disk.

    The data file stores a dictionary with a single top‑level key
    ``users``.  Each entry in ``users`` is keyed by the user's ID as a
    string and contains their preferences, usage statistics and
    subscription state.  If the file does not exist or fails to load, an
    empty structure is returned.
    """
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}}


def save_data(data: Dict[str, Any]) -> None:
    """Save persistent data to disk.

    If saving fails, an error is logged but not raised, because failing
    to persist data should not crash the bot.
    """
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as exc:
        logger.error(f"Failed to save data: {exc}")


def get_user(user_id: int) -> Dict[str, Any]:
    """Retrieve a user's record, creating it with defaults if needed."""
    data = load_data()
    uid = str(user_id)
    user = data.setdefault("users", {}).get(uid)
    if user is None:
        user = DEFAULT_PREFS.copy()
        data["users"][uid] = user
        save_data(data)
    # Fill in any missing keys from DEFAULT_PREFS for backward
    # compatibility with old records.
    changed = False
    for k, v in DEFAULT_PREFS.items():
        if k not in user:
            user[k] = v
            changed = True
    if changed:
        data["users"][uid] = user
        save_data(data)
    return user


def save_user(user_id: int, user_info: Dict[str, Any]) -> None:
    """Persist a user's record to disk."""
    data = load_data()
    data.setdefault("users", {})[str(user_id)] = user_info
    save_data(data)


def is_subscribed(user_info: Dict[str, Any]) -> bool:
    """Return True if the user has an active subscription."""
    sub_until = user_info.get("subscribed_until")
    if sub_until is None:
        return False
    try:
        expiry = datetime.fromisoformat(sub_until)
        return expiry > datetime.utcnow()
    except Exception:
        return False


def is_blocked(user_info: Dict[str, Any]) -> bool:
    """Return True if the user is currently blocked."""
    blocked = user_info.get("blocked_until")
    if blocked is None:
        return False
    try:
        until = datetime.fromisoformat(blocked)
        return until > datetime.utcnow()
    except Exception:
        return False


def add_usage(user_id: int) -> None:
    """Increment a user's usage count and record the current timestamp."""
    user = get_user(user_id)
    user["usage"] = user.get("usage", 0) + 1
    ts_list = user.get("timestamps", [])
    ts_list.append(datetime.utcnow().isoformat())
    user["timestamps"] = ts_list
    save_user(user_id, user)


def add_payment(user_id: int, days: int = 30) -> None:
    """Record a payment and activate a subscription for the specified period.

    A payment adds a timestamp to the user's ``payments`` list, resets
    their usage counter and sets ``subscribed_until`` to ``now + days``.
    """
    user = get_user(user_id)
    # Record the payment
    pay_list = user.get("payments", [])
    pay_list.append(datetime.utcnow().isoformat())
    user["payments"] = pay_list
    # Reset usage and extend subscription
    user["usage"] = 0
    expiry = datetime.utcnow() + timedelta(days=days)
    user["subscribed_until"] = expiry.isoformat()
    save_user(user_id, user)

# ──────────────────────────────────────────────────────────────────────────────
# Logging

def log_event(message: str) -> None:
    """Append a log entry to the log file with a UTC timestamp.

    The log records significant actions such as subscription activations,
    warnings, expirations and blocks.  Use /logs to retrieve this file and
    /clearlogs to erase it.  Errors during logging are silently ignored to
    avoid crashing the bot.
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {message}\n")
    except Exception as exc:
        logging.error(f"Failed to write log entry: {exc}")


# ───────────────────────────────────────────────────────────────────────────────
# Token and referral helpers

def add_tokens(user_id: int, amount: int) -> None:
    """Add a number of tokens to the user's balance.

    If the user's record does not have a 'tokens' field (for
    backward compatibility), it will be created.  Negative amounts
    are ignored.
    """
    if amount <= 0:
        return
    user = get_user(user_id)
    user["tokens"] = int(user.get("tokens", 0)) + amount
    save_user(user_id, user)


def consume_tokens(user_id: int, amount: int) -> bool:
    """Try to subtract a number of tokens from the user's balance.

    Returns True if sufficient tokens were available and were
    successfully deducted, otherwise False and no change is made.  The
    admin always succeeds without deducting tokens.
    """
    if amount <= 0:
        return True
    if user_id == ADMIN_ID:
        # Admin never consumes tokens
        return True
    user = get_user(user_id)
    tokens = int(user.get("tokens", 0))
    if tokens >= amount:
        user["tokens"] = tokens - amount
        save_user(user_id, user)
        return True
    return False


async def is_channel_member(user_id: int) -> bool:
    """Check whether a user is a member of the subscription channel.

    This helper calls the Telegram Bot API to determine if the user has
    joined the channel associated with your CryptoBot subscription.  The
    bot must be an administrator in the channel for this call to
    succeed.  Membership statuses considered as active are
    ``'creator'``, ``'administrator'`` and ``'member'``.  Any other
    status (``'left'``, ``'kicked'``, etc.) is treated as not
    subscribed.  If the check fails (e.g. the bot isn't an admin), the
    function returns False and no error is propagated.
    """
    try:
        member = await bot.get_chat_member(SUBSCRIPTION_CHANNEL, user_id)
        return member.status in ("creator", "administrator", "member")
    except Exception:
        return False


def record_referral(referrer_id: int, new_user_id: int) -> None:
    """Record a referral relationship and award bonus tokens.

    A user can only have one referrer.  If the new user already
    recorded a referrer, no action is taken.  If the referrer already
    has the new user in their partners list, no action is taken.
    Otherwise the referrer earns REFERRAL_BONUS_TOKENS and the new
    user record stores the referrer ID.
    """
    new_user = get_user(new_user_id)
    # Do not allow self‑referrals
    if referrer_id == new_user_id:
        return
    # If the new user already has a referrer, do nothing
    if new_user.get("referrer") is not None:
        return
    # Fetch or create the referrer record
    referrer = get_user(referrer_id)
    # Append the new user to the referrer's partner list if not present
    partners = referrer.get("partners", [])
    if new_user_id in partners:
        return
    partners.append(new_user_id)
    referrer["partners"] = partners
    # Award bonus tokens to the referrer
    referrer["tokens"] = int(referrer.get("tokens", 0)) + REFERRAL_BONUS_TOKENS
    # Persist changes
    save_user(referrer_id, referrer)
    new_user["referrer"] = referrer_id
    save_user(new_user_id, new_user)
    # Notify the referrer and the new user
    try:
        asyncio.create_task(bot.send_message(
            referrer_id,
            f"🎁 Ваш приглашённый пользователь {new_user_id} зарегистрировался. "
            f"Вам начислено {REFERRAL_BONUS_TOKENS} токенов!"
        ))
    except Exception:
        pass
    try:
        asyncio.create_task(bot.send_message(
            new_user_id,
            f"Спасибо, что присоединились по приглашению! Ваша регистрация засчитана"
        ))
    except Exception:
        pass


def add_subscription(user_id: int, plan_key: str = "1m") -> None:
    """Activate or extend a subscription and award tokens.

    The plan_key must be a key in SUBSCRIPTION_PLANS.  The user's
    subscription end date is set to now + plan['days'] (or extended
    if already subscribed to a later date).  The user's usage count
    is reset, and tokens are added according to the plan.
    """
    plan = SUBSCRIPTION_PLANS.get(plan_key)
    if not plan:
        plan = SUBSCRIPTION_PLANS.get("1m")
    user = get_user(user_id)
    # Reset usage so user regains free conversions
    user["usage"] = 0
    # Extend or set subscription
    now = datetime.utcnow()
    current_until = user.get("subscribed_until")
    try:
        current_expiry = datetime.fromisoformat(current_until) if current_until else now
    except Exception:
        current_expiry = now
    new_expiry = current_expiry
    if current_expiry < now:
        # Subscription expired; start from now
        new_expiry = now + timedelta(days=plan.get("days", 30))
    else:
        # Extend from existing expiry
        new_expiry = current_expiry + timedelta(days=plan.get("days", 30))
    user["subscribed_until"] = new_expiry.isoformat()
    # Add tokens
    tokens_to_add = plan.get("tokens", 0)
    user["tokens"] = int(user.get("tokens", 0)) + tokens_to_add
    save_user(user_id, user)
    # Record payment timestamp for analytics and logs
    pay_list = user.get("payments", [])
    pay_list.append(datetime.utcnow().isoformat())
    user["payments"] = pay_list
    save_user(user_id, user)
    log_event(f"Subscription '{plan_key}' activated for user {user_id} (added {tokens_to_add} tokens)")


# ───────────────────────────────────────────────────────────────────────────────
# Subtitle formatting

def format_subtitles(segments, font_name: str, font_size: int, char_limit: int) -> "pysubs2.SSAFile":
    """Create a styled ASS subtitle file from transcription segments.

    Parameters
    ----------
    segments : iterable
        The segments returned by ``faster‑whisper`` during transcription.
    font_name : str
        The name of the font to use.  If the font is not installed on the
        host system, FFmpeg will fall back to a default font.
    font_size : int
        Size of the font in points.  A small value (e.g. 9 or 12) helps
        prevent the subtitles from filling the screen.
    char_limit : int
        Maximum number of characters per line before splitting.  A
        smaller limit is used for portrait videos.

    Returns
    -------
    pysubs2.SSAFile
        An SSA/ASS subtitle object ready to be saved to disk.
    """
    if pysubs2 is None:
        raise RuntimeError(
            "pysubs2 is not installed. Please install it via requirements.txt."
        )
    subs = pysubs2.SSAFile()
    # Define custom style based on user preferences
    style = pysubs2.SSAStyle()
    style.fontname = font_name
    style.fontsize = font_size
    style.bold = True
    style.italic = False
    style.underline = False
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)  # white text
    style.secondarycolor = pysubs2.Color(0, 0, 0, 0)
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
    style.backcolor = pysubs2.Color(0, 0, 0, 96)  # subtle semi‑transparent backdrop
    style.outline = 3
    style.shadow = 0
    style.marginl = 40
    style.marginr = 40
    # Position subtitles near the bottom of the frame.  Increase the
    # vertical margin so text stays out of the centre.  A larger
    # margin is used by default to avoid covering important content.
    style.marginv = 100
    style.alignment = 2  # bottom-centre
    subs.styles["UserStyle"] = style
    # Build subtitle events with line splitting
    for segment in segments:
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        text = segment.text.strip()
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) > char_limit:
                lines.append(current_line.strip())
                current_line = word
            else:
                current_line += " " + word
        if current_line:
            lines.append(current_line.strip())
        # Join lines with \N (ASS newline).  Use raw string to avoid
        # escaping issues on Windows.
        subtitle_text = "\\N".join(lines)
        event = pysubs2.SSAEvent(
            start=start_ms, end=end_ms, text=subtitle_text, style="UserStyle"
        )
        subs.events.append(event)
    return subs


# ───────────────────────────────────────────────────────────────────────────────
# Video processing

def run_ffmpeg(cmd: list[str], cwd: Path | None = None) -> None:
    """Run an FFmpeg command and raise an exception on failure.

    The ``cwd`` parameter ensures that relative subtitle filenames resolve
    correctly on platforms such as Windows, where absolute paths
    containing colons can confuse the ``ass`` filter【65214271342919†L260-L297】.
    """
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd) if cwd else None
    )
    if result.returncode != 0:
        logging.error(result.stderr.decode(errors="ignore"))
        raise RuntimeError(f"FFmpeg exited with code {result.returncode}")


def detect_video_orientation(video_path: Path) -> str:
    """Detect the orientation of a video using ffprobe.

    Returns 'portrait' if height > width, 'landscape' otherwise.  If
    detection fails, returns 'landscape' as a safe default.
    """
    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            dims = proc.stdout.decode().strip().split(",")
            if len(dims) >= 2:
                width, height = int(dims[0]), int(dims[1])
                return "portrait" if height > width else "landscape"
    except Exception:
        pass
    return "landscape"


def process_video(video_path: str, user_info: Dict[str, Any]) -> str:
    """Process a video file based on a user's preferences.

    Steps:
      1. Extract audio and transcribe with Whisper.
      2. Generate subtitles if enabled.
      3. Upscale the video to Full HD or 4K using Real‑ESRGAN if available.
      4. Apply colour and sharpening filters and overlay subtitles (if any).
      5. Save the processed video and return its filename.

    The function uses a temporary directory for intermediate files.  The
    final video is copied next to the original input file with the
    suffix ``_processed.mp4``.
    """
    # Load Whisper model lazily
    global _whisper_model
    if WhisperModel is None:
        raise RuntimeError(
            "faster_whisper is not installed. Please install it via requirements.txt."
        )
    if _whisper_model is None:
        device = "cuda" if False else "cpu"
        _whisper_model = WhisperModel(
            "medium",
            device=device,
            compute_type="float16" if device == "cuda" else "float32",
        )
    # Prepare user preferences
    orientation_pref = user_info.get("orientation", "auto")
    font_label = user_info.get("font", DEFAULT_PREFS["font"])
    font_name = AVAILABLE_FONTS.get(font_label, font_label)
    font_size = int(user_info.get("font_size", DEFAULT_PREFS["font_size"]))
    subtitles_enabled = bool(user_info.get("subtitles", True))
    # Desired output quality: '1080', '2k', or '4k'.  If the user has
    # selected 4K but does not have an active subscription, processing
    # will fall back to 1080p and a warning will be sent in handle_video.
    quality = user_info.get("quality", "1080")
    # Determine orientation: auto uses ffprobe to detect
    src_path = Path(video_path)
    orientation = (
        detect_video_orientation(src_path) if orientation_pref == "auto" else orientation_pref
    )
    # Character limits for splitting lines: use shorter lines for portrait
    char_limit = 15 if orientation == "portrait" else 32
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Copy input video into temp directory
        tmp_video = tmpdir_path / src_path.name
        shutil.copy(src_path, tmp_video)
        # Extract mono 16 kHz audio for transcription
        audio_path = tmpdir_path / "audio.wav"
        cmd_extract_audio = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        run_ffmpeg(cmd_extract_audio)
        # Transcribe audio
        logging.info("Transcribing audio…")
        segments, info = _whisper_model.transcribe(str(audio_path), beam_size=5)
        # Generate subtitles if enabled
        ass_path: Path | None = None
        if subtitles_enabled:
            subs = format_subtitles(segments, font_name=font_name, font_size=font_size, char_limit=char_limit)
            ass_path = tmpdir_path / "subtitles.ass"
            subs.save(str(ass_path))
        # Determine target resolution based on the desired quality and orientation.
        # Only 1080p, 2K and 4K are supported; 8K has been removed because
        # encoding 8K video reliably requires enormous resources.  If the
        # specified quality is not recognised, default to 1080p.
        logging.info("Upscaling via FFmpeg.")
        if quality == "4k":
            width, height = (3840, 2160) if orientation == "landscape" else (2160, 3840)
        elif quality == "2k":
            width, height = (2560, 1440) if orientation == "landscape" else (1440, 2560)
        else:
            width, height = (1920, 1080) if orientation == "landscape" else (1080, 1920)
        # Build the scaling filter
        scale_filter = f"scale={width}:{height}:flags=lanczos"
        # Build the complete filter chain: scaling, colour/sharpness, optional subtitles
        vf_chain = f"{scale_filter},{FFMPEG_FILTERS}"
        if subtitles_enabled and ass_path is not None:
            vf_chain = f"{vf_chain},ass={ass_path.name}"
        # Produce the final video in one FFmpeg invocation.  Use the
        # original video as input and the extracted audio to preserve
        # synchronisation.
        final_path = tmpdir_path / "processed.mp4"
        cmd_final = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_video),
            "-i",
            str(audio_path),
            "-vf",
            vf_chain,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(final_path),
        ]
        run_ffmpeg(cmd_final, cwd=tmpdir_path)
        # Copy the result next to the original file
        output_path = src_path.with_name(src_path.stem + "_processed.mp4")
        shutil.copy(final_path, output_path)
        logging.info(f"Processing complete: {output_path}")
        return str(output_path)


# ───────────────────────────────────────────────────────────────────────────────
# Bot setup

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
_whisper_model = None  # cache for WhisperModel


@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message) -> None:
    """Send a welcome or help message to the user."""
    # Handle referral parameter if present.  When the user opens the bot via a
    # link like https://t.me/<botname>?start=ref<id>, the parameter after
    # /start is available via message.get_args().  We record the referral
    # before sending the welcome message.
    args = message.get_args()
    if args and args.startswith("ref"):
        ref_id_str = args[3:]
        try:
            ref_id = int(ref_id_str)
            record_referral(ref_id, message.from_user.id)
        except Exception:
            pass

    txt = (
        "Привет!\n"
        "Отправь мне видео, и я автоматически распознаю речь, добавлю субтитры,\n"
        "улучшу цвет и резкость и увеличу разрешение до 1080p (или 4K при наличии Real‑ESRGAN).\n"
        f"У каждого пользователя есть {FREE_LIMIT} бесплатных генераций, после чего нужна подписка.\n\n"
        "Команды:\n"
        "  /menu — открыть меню действий и настроек\n"
        "  /status — показать ваш личный кабинет\n"
        "  /subscribe — информация о подписке\n"
        "  /help — вывести эту справку\n"
    )
    if message.from_user.id == ADMIN_ID:
        txt += (
            "\nАдмин‑команды:\n"
            "  /users — список пользователей\n"
            "  /setsub <user_id> [<дней>] — выдать подписку\n"
            "  /resetusage <user_id> — сбросить счётчик\n"
            "  /stats — посмотреть статистику\n"
            "  /block <user_id> [<часов>] — заблокировать\n"
            "  /unblock <user_id> — разблокировать\n"
        )
    await message.reply(txt, reply_markup=MAIN_REPLY_KB)


@dp.message_handler(commands=["menu", "settings"])
async def show_menu(message: types.Message) -> None:
    """Display the main menu with top‑level actions.

    The menu is structured into large single‑row buttons for better
    usability.  Regular users see options for settings, profile and
    subscription.  Administrators get additional management actions.
    """
    rows = []
    # Main actions for all users
    rows.append([
        InlineKeyboardButton(text="🎬 Отправить видео", callback_data="menu_sendvideo"),
    ])
    rows.append([
        InlineKeyboardButton(text="🔧 Настройки", callback_data="menu_settings"),
    ])
    rows.append([
        InlineKeyboardButton(text="👤 Профиль", callback_data="menu_profile"),
    ])
    rows.append([
        InlineKeyboardButton(text="💳 Подписка", callback_data="menu_subscribe"),
    ])
    rows.append([
        InlineKeyboardButton(text="🤝 Партнёрка", callback_data="menu_referral"),
    ])
    rows.append([
        InlineKeyboardButton(text="🌐 Проекты", callback_data="menu_projects"),
    ])
    rows.append([
        InlineKeyboardButton(text="ℹ️ Помощь", callback_data="menu_help"),
    ])
    # Admin actions shown only to the administrator
    if message.from_user.id == ADMIN_ID:
        rows.append([
            InlineKeyboardButton(text="👥 Пользователи", callback_data="menu_admin_users"),
        ])
        rows.append([
            InlineKeyboardButton(text="📈 Статистика", callback_data="menu_admin_stats"),
        ])
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await message.reply("Меню:", reply_markup=keyboard)


@dp.callback_query_handler(lambda c: True)
async def process_callback(callback_query: types.CallbackQuery) -> None:
    """Handle all callback queries from inline keyboards."""
    data = callback_query.data
    user_id = callback_query.from_user.id
    await callback_query.answer()
    # Top‑level menu actions
    if data == "menu_sendvideo":
        await callback_query.message.reply(
            "Пришлите видеофайл как вложение, и я обработаю его. Поддерживаются форматы MP4, MOV и другие."
        )
    elif data == "menu_settings":
        await show_settings_menu(callback_query.message)
    elif data == "menu_profile":
        await show_profile(callback_query.message)
    elif data == "menu_subscribe":
        await send_subscribe_info(callback_query.message)
    elif data == "menu_referral":
        await show_referral_info(callback_query.message)
    elif data == "menu_projects":
        await show_projects(callback_query.message)
    elif data == "menu_help":
        await send_help_with_back(callback_query.message)
    elif data == "menu_admin_users":
        if user_id == ADMIN_ID:
            await show_user_list(callback_query.message)
    elif data == "menu_admin_stats":
        if user_id == ADMIN_ID:
            await show_stats_with_back(callback_query.message)
    # Navigation callbacks
    elif data == "back_menu":
        await show_menu(callback_query.message)
    elif data == "back_settings":
        await show_settings_menu(callback_query.message)
    # User tapped the 'Отправить видео после очистки' button in the watermark
    # instructions.  Remind them to use the main reply keyboard to send
    # their cleaned video file.
    elif data == "send_clean_video":
        await callback_query.message.reply(
            "Теперь отправьте очищенный файл: нажмите кнопку ‘🎬 Отправить видео’ "
            "внизу и загрузите видео, которое вы скачали из watermark‑сервиса."
        )
    # Settings categories
    elif data == "cfg_orientation":
        await show_orientation_menu(callback_query.message)
    elif data == "cfg_font":
        await show_font_menu(callback_query.message)
    elif data == "cfg_size":
        await show_size_menu(callback_query.message)
    elif data == "cfg_subtitles":
        await toggle_subtitles(callback_query)
    elif data == "cfg_quality":
        await show_quality_menu(callback_query.message)
    # Orientation choice
    elif data.startswith("set_orientation:"):
        _, value = data.split(":", 1)
        user = get_user(user_id)
        user["orientation"] = value
        save_user(user_id, user)
        await callback_query.message.reply(f"Ориентация установлена на: {value}")
    # Font choice
    elif data.startswith("set_font:"):
        _, value = data.split(":", 1)
        user = get_user(user_id)
        user["font"] = value
        save_user(user_id, user)
        await callback_query.message.reply(f"Шрифт установлен на: {value}")
    # Size choice
    elif data.startswith("set_size:"):
        _, value = data.split(":", 1)
        user = get_user(user_id)
        user["font_size"] = int(value)
        save_user(user_id, user)
        await callback_query.message.reply(f"Размер шрифта установлен на: {value} pt")
    # Quality choice
    elif data.startswith("set_quality:"):
        _, value = data.split(":", 1)
        # Normalise values to lower‑case for consistency
        value_norm = value.lower()
        if value_norm in {"1080", "2k", "4k"}:
            user = get_user(user_id)
            user["quality"] = value_norm
            save_user(user_id, user)
            await callback_query.message.reply(f"Качество установлено на: {value_norm}")
        else:
            await callback_query.message.reply("Некорректное значение качества.")
    # No matching callback
    else:
        # Admin‑specific callbacks and pagination
        if data.startswith("admin_page:"):
            # Paginate through the user list.  Format: admin_page:<page>
            try:
                _, page_str = data.split(":", 1)
                page = int(page_str)
                await show_user_list_page(callback_query.message, page)
            except Exception:
                pass
        elif data.startswith("admin_user:"):
            # Show admin actions for a specific user.  Format: admin_user:<uid>:<page>
            parts = data.split(":")
            try:
                uid = int(parts[1])
            except Exception:
                return
            await show_admin_user_menu(callback_query.message, uid)
        elif data.startswith("admin_sub_activate:"):
            # Show plan selection for subscription activation
            try:
                _, uid_str = data.split(":", 1)
                uid = int(uid_str)
                if callback_query.from_user.id == ADMIN_ID:
                    await show_plan_menu(callback_query.message, uid, mode="activate")
            except Exception:
                pass
        elif data.startswith("admin_sub_extend:"):
            # Show plan selection for subscription extension
            try:
                _, uid_str = data.split(":", 1)
                uid = int(uid_str)
                if callback_query.from_user.id == ADMIN_ID:
                    await show_plan_menu(callback_query.message, uid, mode="extend")
            except Exception:
                pass
        elif data.startswith("admin_sub_plan:"):
            # Activate or extend subscription according to plan
            # Format: admin_sub_plan:<mode>:<plan_key>:<uid>
            parts = data.split(":")
            if len(parts) < 4:
                return
            _, mode, plan_key, uid_str = parts
            try:
                tgt = int(uid_str)
            except Exception:
                return
            if callback_query.from_user.id != ADMIN_ID:
                return
            plan = SUBSCRIPTION_PLANS.get(plan_key, SUBSCRIPTION_PLANS.get("1m"))
            add_subscription(tgt, plan_key=plan_key)
            # Notify admin
            action_text = "активирована" if mode == "activate" else "продлена"
            await callback_query.message.reply(
                f"Подписка {action_text} для пользователя {tgt} на {plan['days']} дней (\U0001f4b0 {plan['tokens']} токенов)."
            )
            # Notify user
            expiry = datetime.utcnow() + timedelta(days=plan['days'])
            start_date = datetime.utcnow().strftime("%Y-%m-%d")
            end_date = expiry.strftime("%Y-%m-%d")
            try:
                await bot.send_message(
                    tgt,
                    f"✅ Ваша подписка {action_text} на {plan['days']} дней. Срок действия: с {start_date} по {end_date}. Вам начислено {plan['tokens']} токенов."
                )
            except Exception:
                pass
            log_event(f"Admin {callback_query.from_user.id} {action_text} subscription {plan_key} for {tgt}")
            # Return to admin menu for that user
            await show_admin_user_menu(callback_query.message, tgt)
        elif data.startswith("admin_sub_cancel:"):
            # Cancel subscription
            try:
                _, uid_str = data.split(":", 1)
                tgt = int(uid_str)
            except Exception:
                return
            if callback_query.from_user.id != ADMIN_ID:
                return
            user = get_user(tgt)
            user["subscribed_until"] = None
            user["warned_until"] = None
            user["usage"] = 0
            save_user(tgt, user)
            await callback_query.message.reply(f"Подписка для пользователя {tgt} отменена.")
            try:
                await bot.send_message(
                    tgt,
                    "🔻 Ваша подписка была отменена администрацией."
                )
            except Exception:
                pass
            log_event(f"Admin {callback_query.from_user.id} cancelled subscription for {tgt}")
            await show_admin_user_menu(callback_query.message, tgt)
        elif data.startswith("admin_block_duration:"):
            # Block user for a specified number of hours
            try:
                _, hours_str, uid_str = data.split(":", 2)
                hours = int(hours_str)
                tgt = int(uid_str)
            except Exception:
                return
            if callback_query.from_user.id != ADMIN_ID:
                return
            user = get_user(tgt)
            until = datetime.utcnow() + timedelta(hours=hours)
            user["blocked_until"] = until.isoformat()
            # Cancel subscription if blocking indefinitely (arbitrary large hours)
            if hours >= 24 * 36500:
                user["subscribed_until"] = None
                user["warned_until"] = None
            save_user(tgt, user)
            await callback_query.message.reply(f"Пользователь {tgt} заблокирован на {hours} часов.")
            try:
                await bot.send_message(
                    tgt,
                    f"⛔️ Вас заблокировали на {hours} часов. Свяжитесь с администратором для разблокировки."
                )
            except Exception:
                pass
            log_event(f"Admin {callback_query.from_user.id} blocked user {tgt} for {hours} hours")
            await show_admin_user_menu(callback_query.message, tgt)
        elif data.startswith("admin_block:"):
            # Show block duration selection
            try:
                _, uid_str = data.split(":", 1)
                tgt = int(uid_str)
            except Exception:
                return
            if callback_query.from_user.id == ADMIN_ID:
                await show_block_menu(callback_query.message, tgt)
        elif data.startswith("admin_unblock:"):
            try:
                _, uid_str = data.split(":", 1)
                tgt = int(uid_str)
            except Exception:
                return
            if callback_query.from_user.id != ADMIN_ID:
                return
            user = get_user(tgt)
            user["blocked_until"] = None
            save_user(tgt, user)
            await callback_query.message.reply(f"Пользователь {tgt} разблокирован.")
            try:
                await bot.send_message(
                    tgt,
                    "✅ Вы были разблокированы администрацией."
                )
            except Exception:
                pass
            log_event(f"Admin {callback_query.from_user.id} unblocked user {tgt}")
            await show_admin_user_menu(callback_query.message, tgt)
        else:
            # Fallback for unknown admin actions; try legacy admin_action handler
            if data.startswith("admin_action:"):
                parts = data.split(":")
                if len(parts) >= 3:
                    action = parts[1]
                    try:
                        tgt = int(parts[2])
                    except Exception:
                        return
                    # Use existing handlers for backward compatibility
                    if action.startswith("sub"):
                        suffix = action[3:]
                        plan_map = {
                            "30": "1m",
                            "60": "2m",
                            "90": "3m",
                            "365": "1y",
                        }
                        plan_key = plan_map.get(suffix, "1m")
                        plan = SUBSCRIPTION_PLANS.get(plan_key, SUBSCRIPTION_PLANS["1m"])
                        add_subscription(tgt, plan_key=plan_key)
                        await callback_query.message.reply(
                            f"Подписка для пользователя {tgt} активирована на {plan['days']} дней (\U0001f4b0 {plan['tokens']} токенов)."
                        )
                        expiry = datetime.utcnow() + timedelta(days=plan['days'])
                        start_date = datetime.utcnow().strftime("%Y-%m-%d")
                        end_date = expiry.strftime("%Y-%m-%d")
                        try:
                            await bot.send_message(
                                tgt,
                                f"✅ Ваша подписка активирована на {plan['days']} дней. Срок действия: с {start_date} по {end_date}. Вам начислено {plan['tokens']} токенов."
                            )
                        except Exception:
                            pass
                        log_event(f"Admin {callback_query.from_user.id} activated subscription {plan_key} for {tgt}")
                        await show_admin_user_menu(callback_query.message, tgt)
                    elif action.startswith("block"):
                        try:
                            hours = int(action[5:])
                        except Exception:
                            hours = 24
                        user = get_user(tgt)
                        until = datetime.utcnow() + timedelta(hours=hours)
                        user["blocked_until"] = until.isoformat()
                        if hours >= 24 * 36500:
                            user["subscribed_until"] = None
                            user["warned_until"] = None
                        save_user(tgt, user)
                        await callback_query.message.reply(
                            f"Пользователь {tgt} заблокирован на {hours} часов."
                        )
                        try:
                            await bot.send_message(
                                tgt,
                                f"⛔️ Вас заблокировали на {hours} часов. Свяжитесь с администратором для разблокировки."
                            )
                        except Exception:
                            pass
                        log_event(f"Admin {callback_query.from_user.id} blocked user {tgt} for {hours} hours")
                        await show_admin_user_menu(callback_query.message, tgt)
                    elif action == "unblock":
                        user = get_user(tgt)
                        user["blocked_until"] = None
                        save_user(tgt, user)
                        await callback_query.message.reply(f"Пользователь {tgt} разблокирован.")
                        try:
                            await bot.send_message(
                                tgt,
                                "✅ Вы были разблокированы администрацией."
                            )
                        except Exception:
                            pass
                        log_event(f"Admin {callback_query.from_user.id} unblocked user {tgt}")
                        await show_admin_user_menu(callback_query.message, tgt)
                    elif action == "cancel":
                        user = get_user(tgt)
                        user["subscribed_until"] = None
                        user["warned_until"] = None
                        user["usage"] = 0
                        save_user(tgt, user)
                        await callback_query.message.reply(f"Подписка для пользователя {tgt} отменена.")
                        try:
                            await bot.send_message(
                                tgt,
                                "🔻 Ваша подписка была отменена администрацией."
                            )
                        except Exception:
                            pass
                        log_event(f"Admin {callback_query.from_user.id} cancelled subscription for {tgt}")
                        await show_admin_user_menu(callback_query.message, tgt)


async def show_orientation_menu(message: types.Message) -> None:
    """Show a submenu to select video orientation."""
    buttons = [
        [InlineKeyboardButton(text="Авто", callback_data="set_orientation:auto")],
        [InlineKeyboardButton(text="16:9", callback_data="set_orientation:landscape")],
        [InlineKeyboardButton(text="9:16", callback_data="set_orientation:portrait")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_settings")],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.reply("Выберите ориентацию:", reply_markup=keyboard)


async def show_font_menu(message: types.Message) -> None:
    """Show a submenu for selecting a font."""
    rows = []
    for label in AVAILABLE_FONTS.keys():
        rows.append([InlineKeyboardButton(text=label, callback_data=f"set_font:{label}")])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_settings")])
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await message.reply("Выберите шрифт:", reply_markup=keyboard)


async def show_size_menu(message: types.Message) -> None:
    """Show a submenu for selecting font size."""
    buttons = [
        [InlineKeyboardButton(text="Маленький (9 pt)", callback_data="set_size:9")],
        [InlineKeyboardButton(text="Средний (12 pt)", callback_data="set_size:12")],
        [InlineKeyboardButton(text="Большой (16 pt)", callback_data="set_size:16")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_settings")],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.reply("Выберите размер шрифта:", reply_markup=keyboard)


async def show_quality_menu(message: types.Message) -> None:
    """Show a submenu for selecting output quality."""
    buttons = [
        [InlineKeyboardButton(text="1080p", callback_data="set_quality:1080")],
        [InlineKeyboardButton(text="2K", callback_data="set_quality:2k")],
        [InlineKeyboardButton(text="4K", callback_data="set_quality:4k")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_settings")],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.reply("Выберите качество видео:", reply_markup=keyboard)


async def show_settings_menu(message: types.Message) -> None:
    """Display the settings menu with options for orientation, font, size, quality and subtitles."""
    rows = [
        [InlineKeyboardButton(text="🎞 Ориентация", callback_data="cfg_orientation")],
        [InlineKeyboardButton(text="🔤 Шрифт", callback_data="cfg_font")],
        [InlineKeyboardButton(text="🔠 Размер", callback_data="cfg_size")],
        [InlineKeyboardButton(text="⚙️ Качество", callback_data="cfg_quality")],
        [InlineKeyboardButton(text="🚫 Субтитры", callback_data="cfg_subtitles")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await message.reply("Настройки:", reply_markup=keyboard)


async def show_profile(message: types.Message) -> None:
    """Display the user's profile information with a back button."""
    user_id = message.from_user.id
    user = get_user(user_id)
    sub = is_subscribed(user)
    sub_text = "Да" if sub else "Нет"
    # Remaining free conversions or ∞ if subscribed
    remaining = "∞" if sub or user_id == ADMIN_ID else max(0, FREE_LIMIT - user.get("usage", 0))
    # Gather referral stats
    partner_count = len(user.get("partners", []))
    # Build status
    text = (
        f"Ваш ID: {user_id}\n"
        f"Использовано бесплатных генераций: {user.get('usage', 0)} из {FREE_LIMIT}\n"
        f"Оставшиеся бесплатные генерации: {remaining}\n"
        f"Токенов на счёте: {user.get('tokens', 0)}\n"
        f"Подписка активна: {sub_text}\n"
        f"Шрифт: {user.get('font')}\n"
        f"Размер шрифта: {user.get('font_size')} pt\n"
        f"Ориентация: {user.get('orientation')}\n"
        f"Качество: {user.get('quality')}\n"
        f"Субтитры: {'включены' if user.get('subtitles', True) else 'выключены'}\n"
        f"Приглашённых пользователей: {partner_count}\n"
        f"Стоимость подписки: {SUBSCRIPTION_PRICE_USD}$\n"
        f"Ссылка на оплату: {PAYMENT_LINK}"
    )
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
    )
    await message.reply(text, reply_markup=keyboard)


async def show_referral_info(message: types.Message) -> None:
    """Display the user's referral information and link."""
    user_id = message.from_user.id
    user = get_user(user_id)
    # Build referral link using the bot's username
    try:
        me = await bot.get_me()
        bot_username = me.username
    except Exception:
        bot_username = ""
    if bot_username:
        link = f"https://t.me/{bot_username}?start=ref{user_id}"
    else:
        link = ""
    partner_count = len(user.get("partners", []))
    txt = (
        "Приглашайте друзей и получайте бонусы!\n"
        f"Ваша реферальная ссылка: {link}\n"
        f"Приглашённых пользователей: {partner_count}\n"
        f"За каждого приглашённого вы получаете {REFERRAL_BONUS_TOKENS} токенов."
    )
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
    )
    await message.reply(txt, reply_markup=keyboard)


async def show_projects(message: types.Message) -> None:
    """Display a list of other projects with external links."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🧠 Тест IQ", url="https://t.me/iqmanager1_bot")],
            [InlineKeyboardButton(text="🛒 IT Market", url="https://t.me/Itmarketkz1_bot")],
            [InlineKeyboardButton(text="👥 IT Market Group", url="https://t.me/shemizarabotkaonlineg")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")],
        ]
    )
    await message.reply("Наши другие проекты:", reply_markup=keyboard)


async def send_subscribe_info(message: types.Message) -> None:
    """Send subscription information with a back button."""
    if message.from_user.id == ADMIN_ID:
        await message.reply("Вы администратор и не нуждаетесь в подписке.")
        return
    user = get_user(message.from_user.id)
    if is_subscribed(user):
        await message.reply(
            "Ваша подписка уже активна. Вы можете отправлять видео без ограничений.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
            ),
        )
        return
    # Construct subscription info with multiple plan options and contact information.
    text_lines = [
        f"Для продолжения после {FREE_LIMIT} бесплатных генераций необходимо оформить подписку.",
        "\nДоступные тарифы:",
    ]
    # List all plans with price and token allocation
    for key, plan in SUBSCRIPTION_PLANS.items():
        # Human-friendly plan name
        if key == "1m":
            name = "1 месяц"
        elif key == "2m":
            name = "2 месяца"
        elif key == "3m":
            name = "3 месяца"
        elif key == "1y":
            name = "1 год"
        else:
            name = key
        price = plan.get("price")
        tokens = plan.get("tokens")
        text_lines.append(f"{name}: {price}$ → {tokens} токенов")
    text_lines.extend(
        [
            "\nОплата производится через наш канал подписки:",
            f"{PAYMENT_LINK}",
            "После оплаты напишите администратору для активации.",
            "Техническая поддержка: @Mi1Shell",
            "Вопросы по оплате: @MikaHarpier",
        ]
    )
    text = "\n".join(text_lines)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
    )
    await message.reply(text, reply_markup=keyboard)


async def send_help_with_back(message: types.Message) -> None:
    """Send help text with a back button."""
    # Reuse the welcome/help text generation
    txt = (
        "Привет!\n"
        "Отправь мне видео, и я автоматически распознаю речь, добавлю субтитры,\n"
        "улучшу цвет и резкость и увеличу разрешение до 1080p (или 4K при наличии Real‑ESRGAN).\n"
        f"У каждого пользователя есть {FREE_LIMIT} бесплатных генераций, после чего нужна подписка.\n\n"
        "Команды:\n"
        "  /menu — открыть меню действий и настроек\n"
        "  /status — показать ваш личный кабинет\n"
        "  /subscribe — информация о подписке\n"
        "  /help — вывести эту справку\n"
        "\nПоддержка:\n"
        "  Техническая поддержка: @Mi1Shell\n"
        "  Вопросы по оплате: @MikaHarpier\n"
    )
    if message.from_user.id == ADMIN_ID:
        txt += (
            "\nАдмин‑команды:\n"
            "  /users — список пользователей\n"
            "  /setsub <user_id> [<дней>] — выдать подписку\n"
            "  /resetusage <user_id> — сбросить счётчик\n"
            "  /stats — посмотреть статистику\n"
            "  /block <user_id> [<часов>] — заблокировать\n"
            "  /unblock <user_id> — разблокировать\n"
        )
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
    )
    await message.reply(txt, reply_markup=keyboard)


async def show_user_list(message: types.Message) -> None:
    """Display the list of users for the admin with a back button."""
    if message.from_user.id != ADMIN_ID:
        return
    data = load_data()
    users = data.get("users", {})
    # Delegate to paginated view starting from page 0
    await show_user_list_page(message, page=0)


async def show_user_list_page(message: types.Message, page: int = 0) -> None:
    """Show a paginated list of users to the admin with navigation buttons.

    Each page displays up to 8 users with an action button to manage
    them.  Navigation controls allow the admin to move between pages.
    """
    if message.from_user.id != ADMIN_ID:
        return
    data = load_data()
    users = data.get("users", {})
    user_ids = list(users.keys())
    # Sort numerically for consistency
    try:
        user_ids = sorted(user_ids, key=lambda x: int(x))
    except Exception:
        user_ids = sorted(user_ids)
    page_size = 8
    total = len(user_ids)
    if total == 0:
        await message.reply("Нет данных о пользователях.", reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
        ))
        return
    # Clamp page
    max_page = (total - 1) // page_size
    page = max(0, min(page, max_page))
    start = page * page_size
    end = start + page_size
    rows = []
    for uid_str in user_ids[start:end]:
        info = users[uid_str]
        usage = info.get("usage", 0)
        sub = "✅" if is_subscribed(info) else "❌"
        blocked = "⛔️" if is_blocked(info) else "✅"
        display = f"ID {uid_str} | {usage}/{FREE_LIMIT} | Подписка: {sub} | Блок: {blocked}"
        rows.append([
            InlineKeyboardButton(
                text=display,
                callback_data=f"admin_user:{uid_str}:{page}",
            )
        ])
    # Navigation buttons
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"admin_page:{page-1}"))
    if page < max_page:
        nav_buttons.append(InlineKeyboardButton(text="➡️ Вперёд", callback_data=f"admin_page:{page+1}"))
    # Always add back to menu
    nav_buttons.append(InlineKeyboardButton(text="🏠 Меню", callback_data="back_menu"))
    rows.append(nav_buttons)
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await message.reply(
        f"Пользователи (страница {page+1}/{max_page+1}):", reply_markup=keyboard
    )


async def show_admin_user_menu(message: types.Message, target_id: int) -> None:
    """Display admin actions for a specific user.

    The menu provides options to grant a subscription for various
    durations, block or unblock the user.  After performing an
    action, the admin is returned to the first page of the user list.
    """
    if message.from_user.id != ADMIN_ID:
        return
    info = get_user(target_id)
    usage = info.get("usage", 0)
    sub_status = "✅" if is_subscribed(info) else "❌"
    block_status = "⛔️" if is_blocked(info) else "✅"
    header = (
        f"Пользователь {target_id}\n"
        f"Использовано {usage}/{FREE_LIMIT}\n"
        f"Подписка: {sub_status}\n"
        f"Блокировка: {block_status}\n"
    )
    # Offer a simplified set of actions.  Activate subscription shows a
    # second menu asking for payment confirmation, which then leads to
    # duration selection.  Extend subscription immediately asks for
    # duration.  Cancel subscription, block, and unblock perform their
    # actions directly.
    buttons = [
        [InlineKeyboardButton(text="📲 Активировать подписку", callback_data=f"admin_sub_activate:{target_id}")],
        [InlineKeyboardButton(text="🎟 Продлить подписку", callback_data=f"admin_sub_extend:{target_id}")],
        [InlineKeyboardButton(text="🚫 Отменить подписку", callback_data=f"admin_sub_cancel:{target_id}")],
        [InlineKeyboardButton(text="⛔️ Блокировать", callback_data=f"admin_block:{target_id}")],
        [InlineKeyboardButton(text="✅ Разблокировать", callback_data=f"admin_unblock:{target_id}")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_page:0")],
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.reply(header, reply_markup=keyboard)


# ───────────────────────────────────────────────────────────────────────────────
# Admin helper menus

async def show_plan_menu(message: types.Message, target_id: int, mode: str) -> None:
    """Show a menu of subscription plans for activation or extension.

    Parameters
    ----------
    message : types.Message
        The Telegram message to reply to.
    target_id : int
        The user ID whose subscription is being modified.
    mode : str
        Either "activate" or "extend".  The mode is included in the
        callback data so that the handler knows how to log the action.
    """
    rows = []
    label_map = {
        "1m": "1 месяц",
        "2m": "2 месяца",
        "3m": "3 месяца",
        "1y": "1 год",
    }
    for plan_key, plan in SUBSCRIPTION_PLANS.items():
        label = label_map.get(plan_key, plan_key)
        rows.append([
            InlineKeyboardButton(
                text=f"{label} ({plan['tokens']} токенов)",
                callback_data=f"admin_sub_plan:{mode}:{plan_key}:{target_id}",
            )
        ])
    rows.append([
        InlineKeyboardButton(text="⬅️ Назад", callback_data=f"admin_user:{target_id}:0")
    ])
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    action = "активации" if mode == "activate" else "продления"
    await message.reply(f"Выберите тариф для {action} подписки:", reply_markup=keyboard)


async def show_block_menu(message: types.Message, target_id: int) -> None:
    """Show a menu of blocking durations for the admin."""
    rows = []
    # 24 hours and indefinite options
    rows.append([
        InlineKeyboardButton(text="24 часа", callback_data=f"admin_block_duration:24:{target_id}")
    ])
    rows.append([
        InlineKeyboardButton(text="72 часа", callback_data=f"admin_block_duration:72:{target_id}")
    ])
    rows.append([
        InlineKeyboardButton(text="Бессрочно", callback_data=f"admin_block_duration:876000:{target_id}")
    ])
    rows.append([
        InlineKeyboardButton(text="⬅️ Назад", callback_data=f"admin_user:{target_id}:0")
    ])
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
    await message.reply("Выберите срок блокировки:", reply_markup=keyboard)


async def show_stats_with_back(message: types.Message) -> None:
    """Display extended analytics with a back button for the admin."""
    if message.from_user.id != ADMIN_ID:
        return
    data = load_data()
    users = data.get("users", {})
    total_users = len(users)
    total_conversions = 0
    day_conv = week_conv = month_conv = year_conv = 0
    day_new = week_new = month_new = year_new = 0
    total_payments = 0
    now = datetime.utcnow()
    for uid, info in users.items():
        timestamps = []
        for ts in info.get("timestamps", []):
            try:
                dt = datetime.fromisoformat(ts)
                timestamps.append(dt)
                total_conversions += 1
                diff = now - dt
                if diff <= timedelta(days=1):
                    day_conv += 1
                if diff <= timedelta(weeks=1):
                    week_conv += 1
                if diff <= timedelta(days=30):
                    month_conv += 1
                if diff <= timedelta(days=365):
                    year_conv += 1
            except Exception:
                pass
        if timestamps:
            first = min(timestamps)
            diff_first = now - first
            if diff_first <= timedelta(days=1):
                day_new += 1
            if diff_first <= timedelta(weeks=1):
                week_new += 1
            if diff_first <= timedelta(days=30):
                month_new += 1
            if diff_first <= timedelta(days=365):
                year_new += 1
        total_payments += len(info.get("payments", []))
    report = (
        f"Всего пользователей: {total_users}\n"
        f"Всего конвертаций: {total_conversions}\n"
        f"Всего оплат: {total_payments}\n"
        "\n"
        "Конвертации за последние периоды:\n"
        f"  День: {day_conv}\n"
        f"  Неделя: {week_conv}\n"
        f"  Месяц: {month_conv}\n"
        f"  Год: {year_conv}\n"
        "\n"
        "Новые пользователи за последние периоды:\n"
        f"  День: {day_new}\n"
        f"  Неделя: {week_new}\n"
        f"  Месяц: {month_new}\n"
        f"  Год: {year_new}"
    )
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_menu")]]
    )
    await message.reply(report, reply_markup=keyboard)


async def toggle_subtitles(callback_query: types.CallbackQuery) -> None:
    """Toggle subtitle visibility for the user.

    This callback expects to be invoked via an inline keyboard.  It
    uses the callback's ``from_user`` to identify the user rather than
    the message author (which would be the bot).  After toggling, it
    informs the user and returns to the settings menu.
    """
    user_id = callback_query.from_user.id
    user = get_user(user_id)
    current = bool(user.get("subtitles", True))
    user["subtitles"] = not current
    save_user(user_id, user)
    state = "включены" if user["subtitles"] else "выключены"
    try:
        await callback_query.message.reply(f"Субтитры теперь {state}.")
    except Exception:
        pass
    await show_settings_menu(callback_query.message)


@dp.message_handler(commands=["status", "account"])
async def user_status(message: types.Message) -> None:
    """Display the user's current settings, usage and subscription info."""
    user_id = message.from_user.id
    user = get_user(user_id)
    # Determine subscription status
    sub = is_subscribed(user)
    sub_text = "Да" if sub else "Нет"
    if sub:
        remaining = "∞"
    else:
        remaining = max(0, FREE_LIMIT - user.get("usage", 0))
    # Compose status message
    # Include token and referral information in the status
    partner_count = len(user.get("partners", []))
    txt = (
        f"Ваш ID: {user_id}\n"
        f"Использовано бесплатных генераций: {user.get('usage', 0)} из {FREE_LIMIT}\n"
        f"Оставшиеся бесплатные генерации: {remaining}\n"
        f"Токенов на счёте: {user.get('tokens', 0)}\n"
        f"Подписка активна: {sub_text}\n"
        f"Шрифт: {user.get('font')}\n"
        f"Размер шрифта: {user.get('font_size')} pt\n"
        f"Ориентация: {user.get('orientation')}\n"
        f"Качество: {user.get('quality')}\n"
        f"Субтитры: {'включены' if user.get('subtitles', True) else 'выключены'}\n"
        f"Приглашённых пользователей: {partner_count}\n"
        f"Стоимость подписки: {SUBSCRIPTION_PRICE_USD}$\n"
        f"Ссылка на оплату: {PAYMENT_LINK}\n"
        "\n"
        "Отправьте видео, и я обработаю его. После оплаты сообщите администратору для активации подписки."
    )
    await message.reply(txt)


@dp.message_handler(commands=["referral"])
async def referral_cmd(message: types.Message) -> None:
    """Send the user's referral information when they issue /referral."""
    await show_referral_info(message)


@dp.message_handler(commands=["broadcast"])
async def broadcast_cmd(message: types.Message) -> None:
    """Broadcast a message to all users (admin only).

    Usage: /broadcast <text>
    Sends the text to every user in the database except the admin.
    The admin receives confirmation on completion.
    """
    if message.from_user.id != ADMIN_ID:
        return
    # If the admin replies to a message containing media or text, copy
    # that message to all users.  Otherwise treat the command arguments
    # as the broadcast text.
    data = load_data()
    users = [int(uid) for uid in data.get("users", {})]
    # Remove admin from recipients
    users = [uid for uid in users if uid != ADMIN_ID]
    if message.reply_to_message:
        # Broadcast the original message (with attachments) to all users
        original = message.reply_to_message
        sent = 0
        for uid in users:
            try:
                # copy_message preserves photos, videos, documents and captions
                await bot.copy_message(chat_id=uid, from_chat_id=original.chat.id, message_id=original.message_id)
                sent += 1
            except Exception:
                continue
        await message.reply(f"Рассылка отправлена {sent} пользователям.")
        log_event(f"Admin {message.from_user.id} broadcasted a message copy to {sent} users")
    else:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.reply("Использование: /broadcast <сообщение> или ответьте на сообщение для рассылки.")
            return
        text = parts[1].strip()
        if not text:
            await message.reply("Текст рассылки не может быть пустым.")
            return
        sent = 0
        for uid in users:
            try:
                await bot.send_message(uid, f"📢 Рассылка от администратора:\n\n{text}")
                sent += 1
            except Exception:
                continue
        await message.reply(f"Рассылка отправлена {sent} пользователям.")
        log_event(f"Admin {message.from_user.id} sent broadcast text to {sent} users")


@dp.message_handler(commands=["subscribe"])
async def subscribe_info(message: types.Message) -> None:
    """Inform the user about purchasing a subscription."""
    if message.from_user.id == ADMIN_ID:
        await message.reply("Вы администратор и не нуждаетесь в подписке.")
        return
    user = get_user(message.from_user.id)
    if is_subscribed(user):
        await message.reply("Ваша подписка уже активна. Вы можете отправлять видео без ограничений.")
        return
    # Build subscription pricing information from configured plans
    lines = [
        f"Для продолжения после {FREE_LIMIT} бесплатных генераций необходимо оформить подписку.",
        "Доступные тарифы:",
    ]
    for key, plan in SUBSCRIPTION_PLANS.items():
        # Human‑readable label: 1m → 1 месяц, 2m → 2 месяца, 3m → 3 месяца, 1y → 1 год
        label = {
            "1m": "1 месяц",
            "2m": "2 месяца",
            "3m": "3 месяца",
            "1y": "1 год",
        }.get(key, key)
        tokens = plan.get("tokens", 0)
        days = plan.get("days", 0)
        price = plan.get("price", 0)
        lines.append(f"  • {label}: {tokens} токенов, {days} дней, {price}$")
    lines.append(f"\nОплатите по ссылке: {PAYMENT_LINK}")
    lines.append("После оплаты свяжитесь с администратором для активации подписки.")
    await message.reply("\n".join(lines))


@dp.message_handler(commands=["users", "viewusers"])
async def view_users(message: types.Message) -> None:
    """List all users with their usage and subscription status (admin only)."""
    if message.from_user.id != ADMIN_ID:
        return
    # Delegate to the interactive paginated list
    await show_user_list_page(message, page=0)


@dp.message_handler(commands=["setsub"])
async def set_subscription_cmd(message: types.Message) -> None:
    """Activate a subscription for a user using a plan key (admin only).

    Usage: /setsub <user_id> [plan]
    The plan can be one of: 1m, 2m, 3m, 1y.  Defaults to 1m if omitted.
    """
    if message.from_user.id != ADMIN_ID:
        return
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply("Использование: /setsub <user_id> [план]")
        return
    try:
        target_id = int(parts[1])
    except ValueError:
        await message.reply("Некорректный ID пользователя.")
        return
    plan_key = parts[2] if len(parts) >= 3 else "1m"
    if plan_key not in SUBSCRIPTION_PLANS:
        await message.reply("Некорректный план. Доступные планы: 1m, 2m, 3m, 1y")
        return
    plan = SUBSCRIPTION_PLANS[plan_key]
    add_subscription(target_id, plan_key=plan_key)
    # Notify the user about their new subscription
    start_date = datetime.utcnow().strftime("%Y-%m-%d")
    end_date = (datetime.utcnow() + timedelta(days=plan["days"])).strftime("%Y-%m-%d")
    try:
        await bot.send_message(
            target_id,
            f"✅ Ваша подписка активирована на {plan['days']} дней. Срок действия: с {start_date} по {end_date}. Вам начислено {plan['tokens']} токенов.",
        )
    except Exception as exc:
        logging.error(f"Failed to notify user {target_id} about new subscription: {exc}")
    log_event(f"Admin {message.from_user.id} activated subscription {plan_key} for {target_id}")
    await message.reply(
        f"Подписка для пользователя {target_id} активирована на {plan['days']} дней (\U0001f4b0 {plan['tokens']} токенов)."
    )


@dp.message_handler(commands=["resetusage"])
async def reset_usage_cmd(message: types.Message) -> None:
    """Reset a user's usage counter (admin only)."""
    if message.from_user.id != ADMIN_ID:
        return
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply("Использование: /resetusage <user_id>")
        return
    try:
        target_id = int(parts[1])
    except ValueError:
        await message.reply("Некорректный ID пользователя.")
        return
    user = get_user(target_id)
    user["usage"] = 0
    user["timestamps"] = []
    save_user(target_id, user)
    await message.reply(f"Счётчик использования для {target_id} сброшен.")
    log_event(f"Admin {message.from_user.id} reset usage counter for {target_id}")

# ─────────────────────────────────────────────────────────────────────────────
# Watermark removal information

@dp.message_handler(commands=["watermark"])
async def send_watermark_instructions(message: types.Message) -> None:
    """Send step-by-step instructions for removing Sora/Veo watermarks.

    The message includes a link to an external service that removes watermarks
    from videos.  It guides the user through uploading the watermarked
    video, running the AI-based removal and downloading the clean result
    before returning to the bot.  Use this command or the corresponding
    menu button to obtain the instructions.
    """
    # Compose instructions that reference both the MiniApp and the
    # standalone website.  The MiniApp runs inside Telegram via
    # https://t.me/sorrywatermark_bot/sorrywatermarkcom, while the
    # website is available at https://sorrywatermark.com/.  Users can
    # choose either option to remove the Sora/Veo logo from their
    # videos before uploading them to the bot.  The instructions are
    # numbered to guide the user through the process.  Citations
    # reference the SorryWatermark site for context about file
    # support, AI removal and download times【739077277190039†L30-L34】
    #【739077277190039†L38-L43】【739077277190039†L46-L49】.
    text = (
        "Как удалить водяные знаки из видео перед обработкой:\n\n"
        "1. Откройте сервис SorryWatermark. Вы можете использовать наш\n"
        "   мини‑приложение прямо в Телеграме — нажмите кнопку ниже или\n"
        "   перейдите по ссылке:\n"
        f"   {WATERMARK_MINIAPP_URL}\n"
        "   Также можно воспользоваться полной версией сайта (лучше на\n"
        "   компьютере):\n"
        f"   {WATERMARK_REMOVAL_URL}\n"
        "2. Загрузите ваш видеофайл с водяным знаком — перетащите его\n"
        "   в область загрузки или нажмите, чтобы выбрать файл. Сервис\n"
        "   поддерживает MP4, MOV и WebM до 100 МБ и сразу показывает\n"
        "   превью【739077277190039†L30-L34】.\n"
        "3. Выберите режим AI для автоматического удаления логотипа Sora\n"
        "   или отредактируйте маску вручную. Алгоритм сам определяет\n"
        "   водяной знак и удаляет его с кадра【739077277190039†L38-L43】.\n"
        "4. Скачайте чистое видео в HD или 4K — это занимает лишь\n"
        "   несколько секунд【739077277190039†L46-L49】.\n"
        "5. Вернитесь в бот и нажмите ‘🎬 Отправить видео’, чтобы\n"
        "   загрузить очищенный файл для генерации субтитров, улучшения\n"
        "   изображения и масштабирования.\n\n"
        "Если у вас возникнут вопросы, обращайтесь: @Mi1Shell\n"
        "(техподдержка) или @MikaHarpier (оплата)."
    )

    # Create an inline keyboard with a button linking to the MiniApp.  When
    # pressed, Telegram opens the mini‑app inside the chat.  A second
    # button reminds the user to return and send the video after
    # cleaning.  The second button simply closes the web view; the
    # actual upload is performed by pressing the ‘🎬 Отправить видео’
    # button on the persistent reply keyboard.
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton(
            text="🛡 Открыть MiniApp для удаления знака",
            url=WATERMARK_MINIAPP_URL,
        ),
        InlineKeyboardButton(
            text="🎬 Отправить видео после очистки",
            callback_data="send_clean_video",
        ),
    )

    await message.reply(text, reply_markup=markup)

@dp.message_handler(lambda message: message.text == "🛡 Убрать водяной знак")
async def remove_watermark_button(message: types.Message) -> None:
    """Handle the reply keyboard button for watermark removal.

    It calls the same instructions as the /watermark command.
    """
    await send_watermark_instructions(message)


@dp.message_handler(commands=["addtokens"])
async def add_tokens_cmd(message: types.Message) -> None:
    """Admin-only command to add tokens to a user's account.

    Usage: /addtokens <user_id> <amount>

    The administrator can grant additional tokens to a user without
    affecting their subscription period.  This is useful when a
    subscribed user runs out of tokens but still has time left on
    their subscription.  The command will notify both admin and the
    user about the added tokens and record the action in the log.
    """
    # Only the administrator can use this command
    if message.from_user.id != ADMIN_ID:
        await message.reply("Эта команда доступна только администратору.")
        return
    args = message.get_args().strip().split()
    if len(args) < 2:
        await message.reply("Использование: /addtokens <user_id> <количество>")
        return
    try:
        target_id = int(args[0])
        amount = int(args[1])
    except Exception:
        await message.reply("Некорректные параметры. Укажите ID пользователя и количество токенов.")
        return
    if amount <= 0:
        await message.reply("Количество токенов должно быть положительным.")
        return
    add_tokens(target_id, amount)
    await message.reply(f"✅ Пользователю {target_id} добавлено {amount} токенов.")
    # Notify the user about the added tokens
    try:
        await bot.send_message(
            target_id,
            f"🎁 Вам начислено {amount} токенов! Теперь ваш баланс: {get_user(target_id).get('tokens')}"
        )
    except Exception:
        pass
    log_event(f"Admin {message.from_user.id} added {amount} tokens to user {target_id}")


@dp.message_handler(commands=["block"])
async def block_user_cmd(message: types.Message) -> None:
    """Block a user for a specified number of hours (admin only).

    Usage: /block <user_id> [hours]
    """
    if message.from_user.id != ADMIN_ID:
        return
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply("Использование: /block <user_id> [hours]")
        return
    try:
        target_id = int(parts[1])
    except ValueError:
        await message.reply("Некорректный ID пользователя.")
        return
    hours = 24
    if len(parts) >= 3:
        try:
            hours = int(parts[2])
        except ValueError:
            pass
    user = get_user(target_id)
    until = datetime.utcnow() + timedelta(hours=hours)
    # For very large blocks (e.g., >= 100 years), treat as indefinite
    if hours >= 24 * 36500:
        user["blocked_until"] = (datetime.utcnow() + timedelta(days=36500)).isoformat()
        # Cancel existing subscription and warning flags
        user["subscribed_until"] = None
        user["warned_until"] = None
    else:
        user["blocked_until"] = until.isoformat()
    save_user(target_id, user)
    await message.reply(f"Пользователь {target_id} заблокирован на {hours} часов.")
    # Notify user privately
    try:
        await bot.send_message(target_id, f"⛔️ Вас заблокировали на {hours} часов. Свяжитесь с администратором для разблокировки.")
    except Exception:
        pass
    log_event(f"Admin {message.from_user.id} blocked user {target_id} for {hours} hours")


@dp.message_handler(commands=["unblock"])
async def unblock_user_cmd(message: types.Message) -> None:
    """Unblock a previously blocked user (admin only)."""
    if message.from_user.id != ADMIN_ID:
        return
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply("Использование: /unblock <user_id>")
        return
    try:
        target_id = int(parts[1])
    except ValueError:
        await message.reply("Некорректный ID пользователя.")
        return
    user = get_user(target_id)
    user["blocked_until"] = None
    save_user(target_id, user)
    await message.reply(f"Пользователь {target_id} разблокирован.")
    # Notify user
    try:
        await bot.send_message(target_id, "✅ Вы были разблокированы администрацией.")
    except Exception:
        pass
    log_event(f"Admin {message.from_user.id} unblocked user {target_id}")


@dp.message_handler(commands=["stats", "analytics"])
async def show_stats(message: types.Message) -> None:
    """Display extended analytics to the administrator."""
    if message.from_user.id != ADMIN_ID:
        return
    data = load_data()
    users = data.get("users", {})
    total_users = len(users)
    total_conversions = 0
    day_conv = week_conv = month_conv = year_conv = 0
    day_new = week_new = month_new = year_new = 0
    total_payments = 0
    now = datetime.utcnow()
    for uid, info in users.items():
        # Count conversions
        timestamps = []
        for ts in info.get("timestamps", []):
            try:
                dt = datetime.fromisoformat(ts)
                timestamps.append(dt)
                total_conversions += 1
                diff = now - dt
                if diff <= timedelta(days=1):
                    day_conv += 1
                if diff <= timedelta(weeks=1):
                    week_conv += 1
                if diff <= timedelta(days=30):
                    month_conv += 1
                if diff <= timedelta(days=365):
                    year_conv += 1
            except Exception:
                pass
        # Count new users based on earliest timestamp
        if timestamps:
            first = min(timestamps)
            diff_first = now - first
            if diff_first <= timedelta(days=1):
                day_new += 1
            if diff_first <= timedelta(weeks=1):
                week_new += 1
            if diff_first <= timedelta(days=30):
                month_new += 1
            if diff_first <= timedelta(days=365):
                year_new += 1
        # Count payments
        total_payments += len(info.get("payments", []))
    report = (
        f"Всего пользователей: {total_users}\n"
        f"Всего конвертаций: {total_conversions}\n"
        f"Всего оплат: {total_payments}\n"
        "\n"
        "Конвертации за последние периоды:\n"
        f"  День: {day_conv}\n"
        f"  Неделя: {week_conv}\n"
        f"  Месяц: {month_conv}\n"
        f"  Год: {year_conv}\n"
        "\n"
        "Новые пользователи за последние периоды:\n"
        f"  День: {day_new}\n"
        f"  Неделя: {week_new}\n"
        f"  Месяц: {month_new}\n"
        f"  Год: {year_new}"
    )
    await message.reply(report)


# ──────────────────────────────────────────────────────────────────────────────
# Reply keyboard handlers

@dp.message_handler(lambda m: m.text == "🎬 Отправить видео")
async def reply_send_video(message: types.Message) -> None:
    """Prompt the user to attach a video.

    This handler responds to the quick‑action button in the reply keyboard.
    """
    await message.reply(
        "Пришлите видеофайл как вложение, и я обработаю его. Поддерживаются форматы MP4, MOV и другие."
    )


@dp.message_handler(lambda m: m.text == "🔧 Настройки")
async def reply_settings(message: types.Message) -> None:
    """Show the settings menu when the user taps the settings button."""
    await show_settings_menu(message)


@dp.message_handler(lambda m: m.text == "👤 Профиль")
async def reply_profile(message: types.Message) -> None:
    """Show the profile when the user taps the profile button."""
    await show_profile(message)


@dp.message_handler(lambda m: m.text == "ℹ️ Помощь")
async def reply_help(message: types.Message) -> None:
    """Show help text when the user taps the help button."""
    await send_help_with_back(message)


# ──────────────────────────────────────────────────────────────────────────────
# Log retrieval and management

@dp.message_handler(commands=["logs"])
async def send_logs(message: types.Message) -> None:
    """Send the contents of the log file to the administrator."""
    if message.from_user.id != ADMIN_ID:
        return
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        await message.reply("Логов нет.")
        return
    try:
        # Send as a document so that large logs are delivered reliably
        with open(LOG_FILE, "rb") as f:
            await bot.send_document(
                message.chat.id,
                InputFile(f, filename="logs.txt"),
                caption="Логи",
            )
    except Exception as exc:
        await message.reply(f"Не удалось отправить лог: {exc}")


@dp.message_handler(commands=["clearlogs"])
async def clear_logs(message: types.Message) -> None:
    """Clear the log file (admin only)."""
    if message.from_user.id != ADMIN_ID:
        return
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            pass
        await message.reply("Логи очищены.")
        log_event(f"Admin {message.from_user.id} cleared logs")
    except Exception as exc:
        await message.reply(f"Не удалось очистить логи: {exc}")


@dp.message_handler(commands=["referral"])
async def referral_cmd(message: types.Message) -> None:
    """Handle the /referral command and show the referral information."""
    await show_referral_info(message)


# ──────────────────────────────────────────────────────────────────────────────
# Background tasks

async def subscription_monitor() -> None:
    """Periodically check subscriptions and send warnings or block expired users.

    This task runs in the background and inspects each user's subscription
    status.  If a subscription will expire within the next day and a warning
    has not yet been sent for that expiry, the user receives a reminder.  If
    a subscription has expired, the user is blocked indefinitely until an
    administrator reactivates the subscription.  All actions are logged.
    """
    while True:
        # Sleep between checks to avoid spamming.  One hour is a sensible
        # compromise between responsiveness and load.
        await asyncio.sleep(3600)
        data = load_data()
        users = data.get("users", {})
        now = datetime.utcnow()
        for uid_str, info in users.items():
            try:
                uid = int(uid_str)
            except Exception:
                continue
            # Skip admin user
            if uid == ADMIN_ID:
                continue
            sub_until = info.get("subscribed_until")
            if sub_until:
                try:
                    expiry = datetime.fromisoformat(sub_until)
                except Exception:
                    continue
                # Send warning one day before expiry
                warned_until = info.get("warned_until")
                if expiry - now <= timedelta(days=1):
                    if warned_until is None or warned_until != sub_until:
                        try:
                            await bot.send_message(
                                uid,
                                f"🔔 Ваша подписка истекает {expiry.strftime('%Y-%m-%d')}. Пожалуйста, продлите её, чтобы избежать блокировки.",
                            )
                        except Exception as exc:
                            logging.error(f"Failed to send expiry warning to {uid}: {exc}")
                        info["warned_until"] = sub_until
                        log_event(f"Warning sent to {uid} about subscription expiring {sub_until}")
                        save_user(uid, info)
                # If expired, block the user
                if now >= expiry:
                    info["subscribed_until"] = None
                    info["warned_until"] = None
                    # Block indefinitely (100 years) until reactivated
                    block_until = now + timedelta(days=36500)
                    info["blocked_until"] = block_until.isoformat()
                    save_user(uid, info)
                    try:
                        await bot.send_message(
                            uid,
                            "❌ Ваша подписка истекла, и доступ был заблокирован. Свяжитесь с администратором для продления.",
                        )
                    except Exception as exc:
                        logging.error(f"Failed to notify {uid} about expired subscription: {exc}")
                    log_event(f"Subscription expired for {uid}; user blocked until {block_until.isoformat()}")


@dp.message_handler(content_types=types.ContentType.VIDEO)
async def handle_video(message: types.Message) -> None:
    """Process a video sent by the user."""
    video = message.video
    user_id = message.from_user.id
    user = get_user(user_id)
    # Check if blocked
    if is_blocked(user):
        await message.reply("Вы временно заблокированы и не можете отправлять видео.")
        return
    # Determine desired quality
    quality_pref = user.get("quality", "1080").lower()
    # Check subscription status from stored expiry and channel membership
    subscribed = is_subscribed(user)
    if not subscribed:
        # Additionally treat the user as subscribed if they belong to the
        # subscription channel.  This enables automatic activation when
        # users join the CryptoBot channel.
        if await is_channel_member(user_id):
            subscribed = True
    # Build a mutable copy of the user prefs for processing
    processing_user = dict(user)
    # If the user selected 4K but does not have an active subscription, warn and treat
    # as 1080p.  Subscribers can process 4K (but must spend tokens).  Users without
    # subscription will be downgraded to 1080p before token calculation.
    if user_id != ADMIN_ID and not subscribed and quality_pref == "4k":
        await message.reply(
            "4K доступно только при активной подписке. Ваше видео будет обработано в 1080p."
        )
        processing_user["quality"] = "1080"
        quality_pref = "1080"
    # Determine the base token cost for the selected quality
    cost_tokens = TOKENS_PER_QUALITY.get(processing_user.get("quality", "1080"), 0)
    # Decide whether to consume tokens.  Admin never consumes tokens.
    # For unsubscribed users, the first FREE_LIMIT conversions are free.
    # After that, or for subscribed users, each conversion costs tokens.
    if user_id != ADMIN_ID:
        # Determine if tokens should be consumed for this conversion
        consume = True
        if not subscribed and user.get("usage", 0) < FREE_LIMIT:
            consume = False  # free quota for unsubscribed users
        if consume and cost_tokens > 0:
            balance = int(user.get("tokens", 0))
            # If enough tokens, deduct and proceed
            if balance >= cost_tokens:
                consume_tokens(user_id, cost_tokens)
            else:
                # Try to downgrade the requested quality to fit the available tokens.
                current_quality = processing_user.get("quality", "1080")
                downgraded = False
                # Determine possible downgrades order
                alternatives = []
                if current_quality == "4k":
                    alternatives = ["2k", "1080"]
                elif current_quality == "2k":
                    alternatives = ["1080"]
                # Iterate through alternative qualities
                for alt_quality in alternatives:
                    alt_cost = TOKENS_PER_QUALITY.get(alt_quality, 0)
                    if balance >= alt_cost:
                        await message.reply(
                            f"Недостаточно токенов для выбранного качества. Видео будет обработано в {alt_quality}."
                        )
                        processing_user["quality"] = alt_quality
                        cost_tokens = alt_cost
                        consume_tokens(user_id, cost_tokens)
                        downgraded = True
                        break
                if not downgraded:
                    # No affordable quality available
                    await message.reply(
                        "Недостаточно токенов для обработки видео. Пополните счёт или оформите подписку."
                    )
                    return
    # Notify user that processing has started
    await message.reply("Спасибо! Видео загружено. Я начал обработку — это может занять несколько минут.")
    # Download video to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / (video.file_name or f"video_{video.file_id}.mp4")
        file = await bot.get_file(video.file_id)
        await bot.download_file(file.file_path, destination=tmp_path)
        # Process video asynchronously in executor
        loop = asyncio.get_running_loop()
        try:
            processed_path: str = await loop.run_in_executor(
                None,
                process_video,
                str(tmp_path),
                processing_user,
            )
        except Exception:
            logging.exception("Error during video processing")
            await message.reply("Произошла ошибка при обработке видео. Попробуйте позже.")
            return
        # Send processed video
        try:
            with open(processed_path, "rb") as out_file:
                await bot.send_video(
                    message.chat.id,
                    InputFile(out_file, filename=os.path.basename(processed_path)),
                    caption="Готово! Вот улучшенное видео",
                )
        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
        # Update usage statistics for all non‑admin users.  Usage is
        # incremented regardless of subscription status to enable
        # analytics and enforce free quotas on unsubscribed users.  Admin
        # usage is not tracked.
        if user_id != ADMIN_ID:
            add_usage(user_id)


def main() -> None:
    """Start the Telegram bot."""
    if not BOT_TOKEN or BOT_TOKEN == "":
        raise RuntimeError("Bot token is not set. Set BOT_TOKEN environment variable or edit bot.py.")
    logger.info("Starting bot…")
    # Launch the bot with startup hooks to configure commands and background tasks
    async def on_startup(dp):
        await setup_commands()
        # Start subscription monitor background task.  Use asyncio.create_task
        # instead of dp.loop.create_task, because dp.loop may be None on some
        # platforms when on_startup is invoked.
        asyncio.create_task(subscription_monitor())
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)


async def setup_commands() -> None:
    """Register bot commands for users and the administrator.

    These commands appear in the Telegram client menu (paperclip icon) for
    quick access.  The admin receives additional commands specific to
    management tasks.
    """
    # Commands for all private chats (users and admin)
    default_cmds = [
        BotCommand(command="menu", description="Открыть меню"),
        BotCommand(command="status", description="Показать личный кабинет"),
        BotCommand(command="subscribe", description="Информация о подписке"),
        BotCommand(command="help", description="Справка"),
        BotCommand(command="referral", description="Партнёрская программа"),
        BotCommand(command="broadcast", description="(админ) Рассылка пользователям"),
        BotCommand(command="watermark", description="Инструкция по удалению водяных знаков"),
    ]
    try:
        await bot.set_my_commands(default_cmds, scope=types.BotCommandScopeAllPrivateChats())
    except Exception as exc:
        logging.error(f"Failed to set default commands: {exc}")
    # Additional commands for the admin's private chat
    admin_cmds = [
        BotCommand(command="users", description="Список пользователей"),
        BotCommand(command="setsub", description="Выдать подписку"),
        BotCommand(command="resetusage", description="Сбросить счётчик"),
        BotCommand(command="addtokens", description="Добавить токены"),
        BotCommand(command="stats", description="Статистика"),
        BotCommand(command="block", description="Заблокировать пользователя"),
        BotCommand(command="unblock", description="Разблокировать пользователя"),
        BotCommand(command="logs", description="Просмотреть логи"),
        BotCommand(command="clearlogs", description="Очистить логи"),
        BotCommand(command="broadcast", description="Рассылка пользователям"),
    ]
    try:
        await bot.set_my_commands(admin_cmds, scope=types.BotCommandScopeChat(chat_id=ADMIN_ID))
    except Exception as exc:
        logging.error(f"Failed to set admin commands: {exc}")


if __name__ == "__main__":
    main()