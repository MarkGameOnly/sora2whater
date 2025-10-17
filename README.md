# Sora 4K Video Subtitle Bot

This repository contains a ready‑to‑use Telegram bot for automatically adding
TikTok‑style subtitles to videos, enhancing colour/sharpness, and upscaling
their resolution to Full HD (1080p) or 4K.  It also implements a usage
tracking and subscription system: each user receives a limited number of
free video conversions, after which a paid subscription is required.  The bot
is built with **Python**, **aiogram**, **faster‑whisper**, **FFmpeg**, and
**pysubs2**.

## Features

* ✅ **Automatic Transcription** – Speech is converted to text using the
  excellent [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) model (medium size by default). You can adjust the model size in `bot.py` if you need faster processing or higher accuracy.
* ✅ **TikTok‑Style Subtitles** – Subtitles are rendered with a bold white font and semi‑transparent black background.  The bot automatically detects whether the video is landscape (16:9) or portrait (9:16) and adjusts line lengths accordingly.  You can customise the font family and size via the `SUBTITLE_FONTNAME` and `SUBTITLE_FONTSIZE` environment variables to achieve a distinctive look (for example, Times New Roman or your own installed font).
* ✅ **Colour & Sharpness Enhancement** – A gentle filter chain (`hqdn3d`, `eq`, `unsharp`) cleans up noise, boosts saturation/contrast and sharpens the image.
* ✅ **Full HD / 4K Upscaling** – If [`realesrgan-ncnn-vulkan`](https://github.com/xinntao/Real-ESRGAN) is installed on the host, it will be used to upscale frames to 4K. Otherwise, the bot upscales videos to Full HD (1920×1080) using FFmpeg’s high‑quality Lanczos scaler.
* ✅ **Interactive Menu** – A `/menu` command presents buttons for checking your status, purchasing a subscription and (for the admin) listing users or viewing statistics, making the bot easy to navigate without remembering commands.
* ✅ **Analytics for the Admin** – An administrator command `/stats` produces a summary of total users, total conversions and how many conversions/new users were recorded in the last day, week, month and year.
* ✅ **Usage Tracking & Subscription** – Every user gets a limited number of free video conversions (configurable; default is 10). After reaching this limit, the bot prompts them to subscribe via a crypto payment link. An administrator account has unlimited usage and can manage subscriptions.
* ✅ **Docker Support** – A `Dockerfile` is provided for containerised deployment. The image installs FFmpeg and Python dependencies. You can supply your bot token via the `BOT_TOKEN` environment variable when running the container.

## Getting Started

1. **Create a Bot Token**

   Talk to [@BotFather](https://t.me/BotFather) on Telegram and create a new bot. Copy the token provided – it looks like `1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef`.

2. **Clone or Download** this repository. Place your bot token in one of two ways:

   * Edit `bot.py` and replace the `BOT_TOKEN` constant, **or**
   * Set an environment variable when running: `BOT_TOKEN=your_token python bot.py`.

3. **Install Dependencies**

   Ensure Python 3.10 or higher is installed. Then install required packages and FFmpeg:

   ```sh
   sudo apt update && sudo apt install -y ffmpeg
   pip install -r requirements.txt
   ```

   To enable Real‑ESRGAN upscaling, follow the installation instructions in its
   [GitHub repository](https://github.com/xinntao/Real-ESRGAN). If the `realesrgan-ncnn-vulkan` binary is on your `PATH`, the bot will use it automatically.

4. **Run the Bot**

   ```sh
   python bot.py
   ```

   The bot will start polling Telegram for updates. Send a video to the bot and wait for it to reply with an enhanced version. Processing time depends on the length of the video and your hardware.

## Usage Limits, Tokens and Subscription

This bot enforces fair use with a combination of **free conversions**, **tokens** and **time‑limited subscriptions**:

* **Free Conversions** – Every non‑admin user receives a fixed number of free conversions (default 10).  While free conversions remain, you can generate videos without spending tokens or having a subscription.  Once your free quota is exhausted, further conversions will consume tokens.
* **Tokens** – Each video consumes a number of tokens depending on the selected quality:
  * 1080p – 25 tokens
  * 2K – 50 tokens
  * 4K – 100 tokens
  Tokens are awarded when you purchase a subscription or invite new users via your referral link.  When your token balance is insufficient for the requested quality, the bot automatically downgrades the resolution (4K → 2K → 1080p) or asks you to purchase more tokens.
* **Subscription Plans** – Subscriptions combine access time with a generous bundle of tokens.  Plans include 1 month, 2 months, 3 months and 1 year; see the `/subscribe` command in the bot for current pricing and token amounts.  When you purchase a subscription through our crypto channel, you’ll receive a link to join.  After payment, contact the administrator to activate your subscription.
* **Referrals** – Share your personal referral link (available via `/referral`) to earn bonus tokens.  Each new user who starts the bot via your link credits you with additional tokens.

The administrator (ID configured in `bot.py`) has unlimited conversions and can manage other users’ subscriptions and token balances via admin commands.

You can view your current token balance, free conversions remaining, selected quality and other settings using `/status` or by opening the `/menu` interface.

## Commands

* `/start` or `/help` – Show a welcome message and basic usage instructions.
* `/status` or `/account` – Display your user ID, number of videos processed, remaining free conversions, and subscription status.
* `/subscribe` – Provides payment information and a link to the crypto bot for purchasing a subscription.
* `/menu` – Opens an interactive menu with quick‑action buttons.  For regular users the menu offers status and subscription links; for the admin it includes links to user lists and statistics.

### Administrator Commands

The following commands are available only to the administrator (configured via the `ADMIN_ID` constant):

* `/viewusers` – List all users who have used the bot, showing their usage count and subscription status.
* `/setsub <user_id>` – Mark a user as subscribed.  After payment, run this command to unlock unlimited usage for that user.
* `/resetusage <user_id>` – Reset the usage count for a user (e.g. to grant additional free conversions).
* `/stats` or `/analytics` – Show a breakdown of total users, total conversions and how many conversions or new users were recorded in the last day, week, month and year.
* `/addtokens <user_id> <amount>` – Credit a user with additional tokens (useful when a subscriber runs out of tokens).

## Notes

* User usage data and subscription status are stored in a JSON file (`usage.json`) in the bot directory.  The file is automatically created and updated.
* If you wish to customise subtitle appearance (font, size, colour, alignment), adjust the `format_subtitles` function in `bot.py`.
* The bot does **not** store original videos; they are processed in a temporary directory and deleted afterwards.

## Docker Deployment

Build and run the bot inside a container:

```sh
docker build -t sora-bot .
docker run -e BOT_TOKEN=your_token_here -p 8443:8443 sora-bot
```

If you want to run `realesrgan-ncnn-vulkan` inside the container, you will need to
install it yourself or mount it from the host. Otherwise, the fallback
upscaling method will be used.

## Notes

* This bot does not store any user data. Videos are processed in a temporary
  directory and deleted immediately after processing.
* On first run, the Whisper model will be downloaded (~2 GB for the medium
  model). Subsequent runs reuse the downloaded model from your cache.
* If you wish to customise subtitle appearance (font, size, colour, alignment),
  adjust the `format_subtitles` function in `bot.py`.

## Licence

This project is released under the MIT licence. See [`LICENSE`](LICENSE) for
details.