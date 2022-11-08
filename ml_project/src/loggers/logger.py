import logging
import logging.config

log_conf = {
    "version": 1,
    "formatters": {
        "basic": {
            "format": "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": "ml-ops.log",
            "formatter": "basic",
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["file_handler"],
        },
    },
}


def create_root_loger() -> logging.Logger:
    logging.config.dictConfig(log_conf)
    root_logger = logging.getLogger()
    return root_logger
