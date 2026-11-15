"""Optional console output of the mesqual log records for scripts and notebooks."""

import logging


class ConsoleLogging:
    """Console output of the mesqual log records, for scripts and notebooks.

    The mesqual loggers are `logging.getLogger(__name__)` of each module, with
    no level, no handler and default propagation. An application routes them
    with its own logging setup, for example `logging.basicConfig(level=logging.INFO)`.
    Without any setup, Python prints records of level `WARNING` and above.

    `enable` adds one stream handler to the logger `'mesqual'` and sets its
    level. The records also propagate to the root logger: if the application
    has a root handler too, each record prints twice. Use one of the two.

    Colored output needs the optional package `colorlog` (the extra
    `color`). Without it, the output is plain text.

    Examples:
        >>> import logging
        >>> from mesqual.utils.logging import ConsoleLogging
        >>> ConsoleLogging.enable(logging.DEBUG)
        >>> ConsoleLogging.disable()
    """

    LOGGER_NAME = 'mesqual'
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    _handler: logging.Handler | None = None
    _previous_level: int = logging.NOTSET
    _enabled_level: int = logging.NOTSET

    @classmethod
    def enable(cls, level: int | str = logging.INFO, colored: bool = True) -> None:
        """Print the mesqual records of `level` and above to the console.

        A second call replaces the handler of the first call and keeps the
        level from before the first call for `disable`.

        Args:
            level: Level of the logger `'mesqual'`, for example `logging.DEBUG` or `'INFO'`.
            colored: Color the records by level if `colorlog` is installed.

        Raises:
            ValueError: `level` is no logging level, for example `'debug'`.
                Nothing changes then: a handler of an earlier call stays.
        """
        logger = logging.getLogger(cls.LOGGER_NAME)
        previous_level = logger.level
        logger.setLevel(level)
        if cls._handler is None:
            cls._previous_level = previous_level
        else:
            logger.removeHandler(cls._handler)
        cls._enabled_level = logger.level
        handler = logging.StreamHandler()
        handler.setFormatter(cls._formatter(colored))
        logger.addHandler(handler)
        cls._handler = handler

    @classmethod
    def disable(cls) -> None:
        """Remove the handler of `enable` and restore the level of the logger `'mesqual'`.

        The level from before the first `enable` comes back only if the
        logger still has the level that `enable` set. A level that the
        application set after `enable` stays.
        """
        logger = logging.getLogger(cls.LOGGER_NAME)
        if cls._handler is None:
            return
        logger.removeHandler(cls._handler)
        cls._handler = None
        if logger.level == cls._enabled_level:
            logger.setLevel(cls._previous_level)

    @classmethod
    def _formatter(cls, colored: bool) -> logging.Formatter:
        if colored:
            try:
                import colorlog
            except ImportError:
                pass
            else:
                return colorlog.ColoredFormatter(f'%(log_color)s{cls.FORMAT}', log_colors=cls.LOG_COLORS)
        return logging.Formatter(cls.FORMAT)
