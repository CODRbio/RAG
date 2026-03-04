"""Centralized path management for user-isolated and shared data directories."""

from pathlib import Path

DEFAULT_USER_ID = "default"


def _get_data_root() -> Path:
    from config.settings import settings
    return getattr(settings.path, "data", Path("data")).resolve()


class PathManager:
    """Provides user-scoped and shared filesystem paths; creates dirs on access."""

    DEFAULT_USER_ID = DEFAULT_USER_ID

    @classmethod
    def get_data_root(cls) -> Path:
        return _get_data_root()

    @classmethod
    def get_user_dir(cls, user_id: str = DEFAULT_USER_ID) -> Path:
        user_dir = cls.get_data_root() / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    @classmethod
    def get_user_raw_papers_path(cls, user_id: str = DEFAULT_USER_ID) -> Path:
        path = cls.get_user_dir(user_id) / "raw_papers"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_user_parsed_path(cls, user_id: str = DEFAULT_USER_ID) -> Path:
        path = cls.get_user_dir(user_id) / "parsed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_user_library_path(cls, user_id: str, library_name: str) -> Path:
        """Base dir for a scholar library (contains 'pdfs' subdir)."""
        path = cls.get_user_dir(user_id) / "libraries" / library_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_user_library_pdfs_path(cls, user_id: str, library_name: str) -> Path:
        path = cls.get_user_library_path(user_id, library_name) / "pdfs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_shared_dir(cls) -> Path:
        path = cls.get_data_root() / "shared"
        path.mkdir(parents=True, exist_ok=True)
        return path
