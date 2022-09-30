from utils import *
# from .augmentation import *
from .augmentation_oulu_ts import *
from .prepare_data import *

class FasDataset(Dataset):
    def __init__(self, mode, image_size=112, DATA_ROOT=None, data_path=None, patch_size=48, frame_len=120, val_len=120, augment = None, balance = False):
        super(FasDataset, self).__init__()

        self.augment = augment
        self.mode = mode
        self.balance = balance
        self.data_path = data_path
        self.DATA_ROOT = DATA_ROOT

        self.channels = 3
        self.frame_len = frame_len#96
        self.val_len = val_len
        # self.train_image_path = TRN_IMGS_DIR
        # self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.patch_size = patch_size

        self.pixel_mean = [103.52,116.28,123.675]
        self.Pixel_std = [57.375,57.12,58.395]
        self.scale = 1.0 

        self.set_mode(self.mode)

    def set_mode(self, mode):
        self.mode = mode
        print(mode)
        
        if self.mode == 'val':
            self.val_list = load_val_list(os.path.join(self.DATA_ROOT, self.data_path))
            self.num_data = len(self.val_list)
            print('set dataset mode: val #%s'%self.num_data)

        elif self.mode == 'train':
            self.train_list = load_train_list(os.path.join(self.DATA_ROOT, self.data_path))
            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)
            print('set dataset mode: train #%s'%self.num_data)

            if self.balance:
                self.train_list = transform_balance(self.train_list)

        # print(self.num_data)


    def __getitem__(self, index):

        if self.mode == 'train':
            pos = random.randint(0,len(self.train_list)-1)
            label, videoname = self.train_list[pos]
            if int(label) > 0:
                label = 1
            else:
                label = 0

        elif self.mode == 'val':
            # video
            label, videoname = self.val_list[index]
            if int(label) > 0:
                label = 1
            else:
                label = 0
        
        if self.mode == 'train':
            trn_path = os.path.join(self.DATA_ROOT, videoname)
            # print("frame len : ", self.frame_len)
            image_x = np.zeros((self.frame_len*3, 9, 3, self.patch_size, self.patch_size))
            label_list = np.zeros(self.frame_len*3)
            name_list = []
            for name in os.listdir(trn_path):
                if os.path.isfile(os.path.join(trn_path, name)):
                    name_list.append(os.path.join(trn_path, name))
            frames_total = len(name_list)
            # frames_total = len([name for name in os.listdir(trn_path) if os.path.isfile(os.path.join(trn_path, name))])
            for idx in range(self.frame_len):
                for tmp in range(500):
                    img_idx = np.random.randint(2, frames_total)
                    img_dir = name_list[img_idx]
                    color = cv2.imread(img_dir, 1)
                    if color is not None:
                        break
                # img_idx = np.random.randint(2,frames_total)
                # if idx >= frames_total:
                #     img_idx = np.random.randint(0,frames_total)
                # else:
                #     img_idx = idx
                # img_dir = name_list[img_idx]
                # color = cv2.imread(img_dir, 1)
                color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
                color, nda_0, nda_1 = augumentor_1(color, target_shape=(self.patch_size, self.patch_size, 3))
                # color = np.concatenate(color, axis=0)
                # color = np.transpose(color, (0, 3, 1, 2))
                # color = color.astype(np.float32)
                # color = color.reshape([9, 3, self.patch_size, self.patch_size])

                # nda_0 = np.concatenate(nda_0, axis=0)
                # nda_0 = np.transpose(nda_0, (0, 3, 1, 2))
                # nda_0 = nda_0.astype(np.float32)
                # nda_0 = nda_0.reshape([9, 3, self.patch_size, self.patch_size])

                # nda_1 = np.concatenate(nda_1, axis=0)
                # nda_1 = np.transpose(nda_1, (0, 3, 1, 2))
                # nda_1 = nda_1.astype(np.float32)
                # nda_1 = nda_1.reshape([9, 3, self.patch_size, self.patch_size])

                image_x[idx*3, :, :, :, :] = color
                image_x[idx*3+1, :, :, :, :] = nda_0
                image_x[idx*3+2, :, :, :, :] = nda_1
                label_list[idx*3] = 1
                label_list[idx*3+1] = 0
                label_list[idx*3+2] = 0

            return torch.FloatTensor(image_x), torch.LongTensor(np.asarray(label_list).reshape([-1]))
            
        if self.mode == 'val':
            val_path = os.path.join(self.DATA_ROOT, videoname)
            image_x = np.zeros((self.val_len, 9, 3, self.patch_size, self.patch_size))
            label_list = np.zeros(self.val_len)
            name_list = []
            for name in os.listdir(val_path):
                if os.path.isfile(os.path.join(val_path, name)):
                    name_list.append(os.path.join(val_path, name))
            frames_total = len(name_list)
            # frames_total = len([name for name in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, name))])
            for idx in range(self.val_len): #in range(self.frame_len):
                for tmp in range(500):
                    img_idx = np.random.randint(2, frames_total)
                    img_dir = name_list[img_idx]
                    # img_dir = '%s/%s.jpg' % (val_path, img_idx)
                    color = cv2.imread(img_dir, 1)
                    if color is not None:
                        break
                # img_idx = np.random.randint(2, frames_total)
                # # if idx >= frames_total:
                # #     img_idx = np.random.randint(0,frames_total)
                # # else:
                # #     img_idx = idx
                # # img_dir = val_path + '/%s.jpg'%img_idx
                # img_dir = name_list[img_idx]
                # color = cv2.imread(img_dir, 1)
                color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
                color = augumentor_2(color, target_shape=(self.patch_size, self.patch_size, 3))
                # image = np.concatenate(color, axis=0)
                # image = np.transpose(image, (0, 3, 1, 2))
                # image = image.astype(np.float32)
                # image = image.reshape([9, 3, self.patch_size, self.patch_size])
                image_x[idx, :, :, :, :] = color#image
                label_list[idx] = int(label)
                
            return torch.FloatTensor(image_x), torch.LongTensor(np.asarray(label_list).reshape([-1]))

    def __len__(self):
        return self.num_data
