from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

from mucha.config import load_config
from mucha.discord_bot import MuchaClient


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    load_dotenv()
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Brak DISCORD_TOKEN. Skopiuj .env.example do .env i wklej token bota.")
    cfg = load_config("config.toml")
    bot = MuchaClient(cfg)
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
