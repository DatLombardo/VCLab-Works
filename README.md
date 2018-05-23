VCLab-Works
===============
Summer 2018 Research in Visual Computing Lab

### Contributors  
* Michael Lombardo
* Faisal Qureshi

Workstation
---------

<details><summary>Connection</summary>
<p>
&emsp;&emsp;ssh lombardo@172.24.6.119

</details>


How to Run Yolo - Object Detection
---------
Must be in darkflow-master folder to execute command
```javascript
python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo leaves_jumping.mp4 --gpu 1.0 –saveVideo
```

Required Imports for Scripts
```python
from darkflow.net.build import TFNet
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.15,
    'gpu': 1.0
}
tfnet = TFNet(option)
```
