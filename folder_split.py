import splitfolders
import os
import shutil

input_path = "/home/videorighter/detectron/FACELIP_DATA/annotations"
output_path = "/home/videorighter/detectron/FACELIP_DATA/split_annotations"

splitfolders.ratio(input_path, output= output_path,
                   seed=1337, ratio=(.8, .2), group_prefix=None)

png_path = "/home/videorighter/detectron/FACELIP_DATA/images/"

train_face_lst = os.listdir("/home/videorighter/detectron/FACELIP_DATA/split_annotations/train/face")
train_lip_lst = os.listdir("/home/videorighter/detectron/FACELIP_DATA/split_annotations/train/lip")
val_face_lst = os.listdir("/home/videorighter/detectron/FACELIP_DATA/split_annotations/val/face")
val_lip_lst = os.listdir("/home/videorighter/detectron/FACELIP_DATA/split_annotations/val/lip")

for file_name in train_face_lst:
    os.rename(output_path+"/train/face/"+file_name,
              output_path+"/train/"+file_name)
os.rmdir(output_path+"/train/face")

for file_name in train_lip_lst:
    shutil.move(output_path+"/train/lip/"+file_name,
                output_path+"/train/"+file_name)
os.rmdir(output_path+"/train/lip")

for file_name in val_face_lst:
    shutil.move(output_path+"/val/face/"+file_name,
                output_path+"/val/"+file_name)
os.rmdir(output_path+"/val/face")

for file_name in val_lip_lst:
    shutil.move(output_path+"/val/lip/"+file_name,
                output_path+"/val/"+file_name)
os.rmdir(output_path+"/val/lip")

png_lst = os.listdir(png_path)
png_lst2 = [x[:-4] for x in png_lst]
train_lst = os.listdir(output_path+"/train")
val_lst = os.listdir(output_path+"/val")

for train in train_lst:
    if train[:-4] in png_lst2:
        try:
            shutil.copy(png_path+train[:-4]+".png", output_path+"/train/"+train[:-4]+".png")
        except FileNotFoundError:
            try:
                shutil.copy(png_path+train[:-4]+".jpg", output_path+"/train/"+train[:-4]+".jpg")
            except FileNotFoundError:
                shutil.copy(png_path+train[:-4]+".png", output_path+"/train/"+train[:-4]+".png")

for val in val_lst:
    if val[:-4] in png_lst2:
        try:
            shutil.copy(png_path+val[:-4]+".png", output_path+"/val/"+val[:-4]+".png")
        except FileNotFoundError:
            try:
                shutil.copy(png_path+val[:-4]+".jpg", output_path+"/val/"+val[:-4]+".jpg")
            except FileNotFoundError:
                shutil.copy(png_path+val[:-4]+".png", output_path+"/val/"+val[:-4]+".png")


