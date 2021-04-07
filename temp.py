# import shutil
# import os
# import random

# VAL_NUM = 500

# image_path = '/home/tiankang/wusuowei/data/kaggle/carvana/train'
# mask_path = '/home/tiankang/wusuowei/data/kaggle/carvana/train_masks'
# val_image_path = '/home/tiankang/wusuowei/data/kaggle/carvana/val'
# val_mask_path = '/home/tiankang/wusuowei/data/kaggle/carvana/val_masks'

# image_list = os.listdir(val_image_path)

# val_idx = sorted(random.sample(range(len(image_list)), VAL_NUM))

# for idx in val_idx:
#     source_image = os.path.join(image_path, image_list[idx])
#     target_image = os.path.join(val_image_path, image_list[idx])
#     shutil.move(target_image, source_image)

#     source_mask = os.path.join(mask_path, image_list[idx].replace('.jpg', '_mask.gif'))
#     target_mask = os.path.join(val_mask_path, image_list[idx].replace('.jpg', '_mask.gif'))
#     shutil.move(target_mask, source_mask)
