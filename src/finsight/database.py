from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from .config import Settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        get_database_url(), pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )


def get_database_url() -> URL:
    settings = Settings()
    return URL.create(
        drivername="postgresql+psycopg",
        username=settings.db_user,
        password=settings.db_password.get_secret_value(),
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
