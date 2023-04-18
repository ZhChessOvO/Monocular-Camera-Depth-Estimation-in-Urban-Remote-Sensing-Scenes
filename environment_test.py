import platform
import time

if __name__ == '__main__':
    print(platform.python_version())
    print(time.asctime(time.localtime(time.time())))
