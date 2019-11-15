from skimage import io, transform
import tensorflow as tf
import numpy as np
import datetime

starttime = datetime.datetime.now()

# path = "./testandtraindata/airplanes/image_0002.jpg"
path = "./timg.jpg"

pic_dict = {0: 'dragonfly', 100: 'ceiling_fan', 101: 'snoopy', 10: 'windsor_chair', \
            11: 'scissors', 12: 'sea_horse', 13: 'inline_skate', 14: 'ferry', 15: 'gerenuk', 16: 'accordion', \
            17: 'water_lilly', 18: 'cannon', 19: 'menorah', 1: 'lobster', 20: 'brontosaurus', 21: 'chair',
            22: 'starfish', \
            23: 'octopus', 24: 'rhino', 25: 'yin_yang', 26: 'electric_guitar', 27: 'bass', 28: 'wild_cat', 29: 'crab', \
            2: 'dolphin', 30: 'pyramid', 31: 'watch', 32: 'mandolin', 33: 'Faces_easy', 34: 'ant', 35: 'buddha',
            36: 'bonsai', \
            37: 'flamingo', 38: 'wheelchair', 39: 'dollar_bill', 3: 'okapi', 40: 'platypus', 41: 'kangaroo',
            42: 'Motorbikes', \
            43: 'elephant', 44: 'Leopards', 45: 'barrel', 46: 'gramophone', 47: 'butterfly', 48: 'grand_piano', \
            49: 'BACKGROUND_Google', 4: 'revolver', 50: 'crocodile', 51: 'car_side', 52: 'anchor', 53: 'emu', \
            54: 'trilobite', 55: 'hedgehog', 56: 'flamingo_head', 57: 'schooner', 58: 'panda', 59: 'llama', \
            5: 'cougar_body', 60: 'garfield', 61: 'stapler', 62: 'Faces', 63: 'dalmatian', 64: 'binocular', \
            65: 'cougar_face', 66: 'crocodile_head', 67: 'ketch', 68: 'headphone', 69: 'pizza', 6: 'scorpion', \
            70: 'tick', 71: 'pagoda', 72: 'nautilus', 73: 'helicopter', 74: 'strawberry', 75: 'crayfish',
            76: 'euphonium', \
            77: 'metronome', 78: 'stop_sign', 79: 'camera', 7: 'ewer', 80: 'cellphone', 81: 'ibis', 82: 'chandelier', \
            83: 'sunflower', 84: 'beaver', 85: 'joshua_tree', 86: 'lamp', 87: 'umbrella', 88: 'minaret', 89: 'brain', \
            8: 'cup', 90: 'wrench', 91: 'hawksbill', 92: 'saxophone', 93: 'lotus', 94: 'pigeon', 95: 'soccer_ball', \
            96: 'mayfly', 97: 'rooster', 98: 'airplanes', 99: 'stegosaurus', 9: 'laptop'}
# pic_dict = {0:'dragonfly',16:'accordion',52:'anchor',98:'airplanes'}

w = 100
h = 100
c = 3


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h, c))
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    data1 = read_one_image(path)
    data.append(data1)

    saver = tf.train.import_meta_graph('./modal/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./modal/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应分类
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("dict:" + pic_dict[output[i]])

endtime = datetime.datetime.now()
runtime = endtime - starttime
print(runtime)
