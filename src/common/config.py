import os

from dotenv import load_dotenv

load_dotenv()


APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MAX_PRODUCTS = int(os.getenv("MAX_PRODUCTS", "100"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
#The first argument is the environment variable name.
#The second argument is a default value.

