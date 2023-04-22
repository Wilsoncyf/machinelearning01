import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_path):
    img = plt.imread(file_path)
    if len(img.shape) == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # 将彩色图像转换为灰度图像
    img = 1 - img  # 反转颜色：黑底白字变为白底黑字
    img = img.reshape(1, 28, 28, 1)  # 适应模型输入
    return img

def predict_digit(model, img):
    prediction = model.predict(img)
    return np.argmax(prediction)

if __name__ == "__main__":
    model = tf.keras.models.load_model('mnist_model.h5')
    img_path = "/Volumes/data/tensorFlow/2.png"
    img = load_image(img_path)
    digit = predict_digit(model, img)

    print(f"图片中的数字是: {digit}")
