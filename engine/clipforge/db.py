"""SQLite engine and sessions. WAL mode lets the API read while the worker writes."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event, inspect
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from clipforge import models  # noqa: F401  (registers tables)
from clipforge.config import get_settings


class SchemaMismatch(RuntimeError):
    pass


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    engine = create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _record) -> None:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    check_schema(engine, settings.db_path)
    return engine


def check_schema(engine: Engine, path: Path) -> None:
    """Fail at startup, not silently later: a stale table would quietly break the 24-hour cleanup.

    Nothing is migrated automatically, so your data is never changed behind your back.
    """
    inspector = inspect(engine)
    missing = []
    for table in SQLModel.metadata.sorted_tables:
        present = {column["name"] for column in inspector.get_columns(table.name)}
        missing += [f"{table.name}.{c.name}" for c in table.columns if c.name not in present]
    if missing:
        raise SchemaMismatch(
            f"The database at {path} is from an older clipforge version (missing {', '.join(missing)}). "
            "Stop clipforge and move that file somewhere safe, and a new one will be created. "
            "Jobs and clip records in it will no longer be listed."
        )


def session_scope() -> Session:
    return Session(get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    with session_scope() as session:
        yield session
