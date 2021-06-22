import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logging.debug('This message should go to the log file')
print("TESTING PRINT")
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too')