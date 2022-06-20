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
            'filename': os.environ["LOG_PATH"],
            'mode': 'a',
        },
        # 'critical_mail_handler': {
        #     'level': 'CRITICAL',
        #     'formatter': 'error',
        #     'class': 'logging.handlers.SMTPHandler',
        #     'mailhost' : 'localhost',
        #     'fromaddr': 'monitoring@domain.com',
        #     'toaddrs': ['dev@domain.com', 'qa@domain.com'],
        #     'subject': 'Critical error with application name'
        # }
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
        'paramiko':{
            'level': "WARNING"
        }
    } 
}