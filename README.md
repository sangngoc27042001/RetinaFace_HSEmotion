# RetinaFace_HSEmotion
 
## 1. Import and use like a library
```python
from RetinaFace_HSEmotion import get_bboxes_emotion
import cv2

img = cv2.imread('./test/20180720_174416.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bbs_em = get_bboxes_emotion(img)

print(bbs_em)
```
```
[(array([1489, 1616, 1787, 1995]), 'Anger'), (array([ 975, 1389, 1375, 1850]), 'Contempt'), (array([2009, 1634, 2292, 1973]), 'Fear')]
```
## 2 Use the command line
We will use the file [demo.py](./demo.py), to apply the pipeline to a specific folder:
```bash
python demo.py --input './test' --output './out'
```
 The result will show the status of each image in the input folder:
 ```
 .\test\20180720_174416.jpg [DONE] [3 faces detected]
.\test\OIP.jpeg [DONE] [1 face detected]
All images have been saved in folder: ./out
 ```