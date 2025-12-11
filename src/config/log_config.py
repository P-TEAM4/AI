"""Uvicorn logging configuration with KST timezone"""

from datetime import datetime, timezone, timedelta


# KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))


class KSTFormatter:
    """Custom formatter to use KST timezone for uvicorn logs"""

    def __init__(self, fmt: str):
        self.fmt = fmt

    def formatTime(self, record):
        dt = datetime.fromtimestamp(record.created, tz=KST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        record.asctime = self.formatTime(record)
        return self.fmt % record.__dict__


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "src.config.log_config.KSTFormatter",
            "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "access": {
            "()": "src.config.log_config.KSTFormatter",
            "fmt": '%(asctime)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}
