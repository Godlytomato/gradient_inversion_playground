from observer.logging.logging_def import logging_def

from observer.logging.implementations.file import file_logger
from observer.logging.implementations.database import database_logger
from observer.logging.implementations.none import none_logger

logging_registry = {
    "file": logging_def(
        name="file",
        function=file_logger,
        requires=set(),
    ),

    "database": logging_def(
        name="database",
        function=database_logger,
        requires=set(),
    ),
    "none": logging_def(
        name="none",
        function=none_logger,
        requires=set(),
    ),
}