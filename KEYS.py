import os
from dotenv import load_dotenv

load_dotenv()

def DEEPSEEK_KEY():
    return os.getenv('DEEPSEEK_KEY')

def GOOG_CSE_ID():
    return os.getenv('GOOG_CSE_ID')

def GOOG_KEY():
    return os.getenv('GOOG_KEY')

if __name__ == '__main__':
    print(f'GOOG_KEY > {GOOG_KEY()}')