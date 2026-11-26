from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8-sig",
        hide_input_in_errors=True,
    )

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: SecretStr
