# YOLOv8_tools

integrate wandb callback with yolov8

use code

```python
from ultralytics import YOLO 
from wandb_callback import callbacks 

model = YOLO('yolov8s.pt') 

for event,func in callbacks.items():
    model.add_callback(event,func)

results = model.train(data = 'coco128.yaml', epochs = 400, workers=8)
```
