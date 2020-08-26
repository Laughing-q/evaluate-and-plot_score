import os
import json
from pathlib import Path

if not os.path.exists('image_id'):
    os.mkdir('image_id')
    
img_dir = 'your_annotation_path'

json_file = os.path.join(img_dir, "instances_val.json")
with open(json_file) as f:
    val_targets = json.load(f)


record = {}
# hw = np.loadtxt(os.path.join(img_dir, 'hw.txt'), dtype=np.long)
for i, image in enumerate(val_targets['images']):
    filename = Path(image['file_name']).stem
    # height, width = cv2.imread(filename).shape[:2]
    # width, height = Image.open(filename).size
    idx = image['id']
    # with open(filename.replace('jpg', 'txt').replace('png', 'txt'), 'a') as f:
    #     f.write(str(idx))
    record[filename] = idx

print(len(record))
f = 'image_id/val_id.json'
with open(f, 'w') as file:
    json.dump(record, file)
