import os
from dotenv import load_dotenv

load_dotenv()

def DEEPSEEK_KEY():
    return os.getenv('DEEPSEEK_KEY')

def GOOG_CSE_ID():
    return os.getenv('GOOG_CSE_ID')

def GOOG_KEY():
    return os.getenv('GOOG_KEY')

def ACCESS_KEY():
    return os.getenv('ACCESS_KEY')

def FLASK_SECRET():
    return os.getenv('FLASK_SECRET')

if __name__ == '__main__':
    print(f'ACCESS_KEY > {ACCESS_KEY()}')