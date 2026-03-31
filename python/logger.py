import logging
import os
from datetime import datetime

class LoggerGenerator:
    _logger_cache = {}

    @staticmethod
    def get_logger(log_dir, name=None, console_output=True):
        if name is None:
            import inspect
            caller_frame = inspect.currentframe().f_back
            name = caller_frame.f_globals.get("__name__", "unknown")

        if name in LoggerGenerator._logger_cache:
            return LoggerGenerator._logger_cache[name]

        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  
        logger.propagate = False  

        if logger.hasHandlers():
            logger.handlers.clear()
        log_filename = os.path.join(
            log_dir,
            f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        if console_output:
            console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        if console_output:
            logger.addHandler(console_handler)

        LoggerGenerator._logger_cache[name] = logger

        return logger


if __name__ == "__main__":
    log_directory = "./logs"

    logger = LoggerGenerator.get_logger(log_directory, name="model", console_output=True)

    logger.debug("这是 debug 信息")
    logger.info("这是 info 信息")
    logger.warning("这是 warning 信息")
    logger.error("这是 error 信息")
    logger.critical("这是 critical 信息")