from util.io import IOUtils

tk = IOUtils(is_cuda=True)
fp = '/home/x/data/HIMYM/01.jpg'

var_img = tk.load_image(fp, as_var=False)
import matplotlib.pyplot as plt
plt.imshow(var_img)
plt.show()
print(var_img.size())
print(var_img.is_cuda)
