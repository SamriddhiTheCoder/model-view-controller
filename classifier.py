import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
from sklearn.linear_model import LogisticRegression

X = np.load("image.npz")["arr_0"]
Y = pd.read_csv("labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train,  y_test = train_test_split(X, Y, test_size=500, train_size=3500, random_state=9)

clf = LogisticRegression(solver="saga", multi_class="multinomial")
#clf.fit(x_train, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resized((22, 30), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    
    image_bw_resized_inverted_scaled = np.asanyarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 600)
    test_pred = clf.predict(test_sample)
    return test_pred