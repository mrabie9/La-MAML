import torch
import sys
import subprocess
print('torch', torch.__version__)
print('torch cuda', torch.version.cuda)
print('cudnn', torch.backends.cudnn.version())
print('python', sys.version)
print('device', torch.cuda.get_device_name(0),
      'capability', torch.cuda.get_device_capability(0))
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
