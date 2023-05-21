import sys

if sys.platform.startswith('win'):
    # 当前平台为 Windows
    print('This is Windows')
elif sys.platform.startswith('linux'):
    # 当前平台为 Linux
    print('This is Linux')
else:
    # 其他平台
    print('Unknown platform')
