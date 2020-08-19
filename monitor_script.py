import os
import psutil
import sys

pid = int(sys.argv[1])
py = psutil.Process(pid)
print(py)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)