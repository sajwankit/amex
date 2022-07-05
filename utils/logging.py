import logging
import os, sys

LOGGING_CONFIG = { 
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': { 
        'console': { 
            'level': os.environ["LOG_LEVEL"],
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'file': {
            'level': os.environ["LOG_LEVEL"],
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': f"""{os.environ["PROJECT_DIR"]}logs/{os.environ["LOG_FILENAME"]}""",
            # 'mode': 'a',
            'mode': 'w'
        },
    },
    'loggers': { 
        '': { 
            'handlers': ['console', 'file'],
            'level': os.environ["LOG_LEVEL"],
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['console', 'file'],
            'level': os.environ["LOG_LEVEL"],
            'propagate': False
        },
        'mustlog': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'numba':{
            'level': "WARNING"
        }
    } 
}
