import cv2
from insightface.app import FaceAnalysis
import numpy as np
import glob
import pandas as pd


## Find faces, estimate their gender and age

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

img = cv2.imread("group4.jpg")
faces = app.get(img)
for f in faces:
    print(f.bbox, f.det_score, f.kps.shape, f.normed_embedding.shape)
    
cv2.imwrite("out.jpg", app.draw_on(img, faces))

## See if the pictures belong to the same person or not (> 0.4 indicates that it's likely to be the same person)
emb = lambda p: app.get(cv2.imread(p))[0].normed_embedding
print(np.dot(emb("yd1.jpg"), emb("yd2.jpg")))

## Find reference person in a group of people
me = emb("yd1.jpg")  # the ref pic
for f in faces:      # faces from group.jpg
    sim = float(np.dot(me, f.normed_embedding))
    print(f.bbox.astype(int), round(sim, 3))
    if sim > 0.4:
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(img, f"ME {sim:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.imwrite("found_me.jpg", img)

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); app.prepare(ctx_id=-1)
paths = sorted(glob.glob("people/*/*.jpg"))
E = np.array([app.get(cv2.imread(p))[0].normed_embedding for p in paths])
S = E @ E.T  # every embedding compared with every other = 9x9 similarity matrix
names = [f"{Path(p).parent.name}/{Path(p).stem}" for p in paths]
print(pd.DataFrame(S, index=names, columns=names).round(2))