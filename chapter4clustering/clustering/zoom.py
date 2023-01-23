# create a pre-prcoessing procedure to zoom into same FOV as 2018 images with interpolation
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt

path = '/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/project/pathways/live/Transferability/emily_phd_images/'
path_2018 = path + '2018/2018/2018/2018/2018/'
path_2021 = path + 'census_2021/2021/'

intersection = pd.read_csv('/home/emily/phd/003_image_matching/census/intersection.csv')[['idx_x', 'idx_y']]

for i in [1,10,100,1000]:
    img_2018 = Image.open(path_2018 + str(intersection.iloc[i]['idx_x']) + '_d.png')
    img_2021 = Image.open(path_2021 + str(intersection.iloc[i]['idx_y']) + '_d.png')

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img_2018)
    #axarr[1].imshow(img_2021)
    axarr[1].imshow(preprocess(img_2021))
    plt.savefig('/home/emily/phd/003_image_matching/census/interpolate_%s.png' % i)

def preprocess(img):
    (left, upper, right, lower) = (137, 132, 505, 500)
    # Here the image "im" is cropped and assigned to new variable im_crop
    img_crop = img.crop((left, upper, right, lower))
    img_crop = img_crop.resize((640,640),Image.BICUBIC)
    return img_crop

f, axarr = plt.subplots(1,2)
axarr[0].imshow(img_2018)
#axarr[1].imshow(img_2021)
axarr[1].imshow(preprocess(img_2021))