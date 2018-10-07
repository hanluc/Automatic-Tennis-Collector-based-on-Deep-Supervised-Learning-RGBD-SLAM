from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *



#data reader  ----------------------------------------------------------------
class ScienceDataset(Dataset):

    def __init__(self, data_dir, set_name, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()

        self.data_dir = data_dir
        self.set_name = set_name
        self.transform = transform
        self.mode = mode
        self.load_DSB2018(data_dir, set_name)

    def __getitem__(self, index):
        if self.mode in ['train']:
            meta = 0
            multi_mask = self.load_mask(index)
            image = self.load_image(index)

            if self.transform is not None:
                return self.transform(image, multi_mask, index)
            else:
                return image, multi_mask, index

        if self.mode in ['test']:
            image = self.load_image(index)
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.filenames)

    def load_DSB2018(self, data_dir, set_name, config=None):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        print(set_name)
        self.filenames = np.genfromtxt(set_name, dtype=str)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        data_dir = self.data_dir
        name = self.filenames[image_id]
        
        img = cv2.imread(os.path.join(data_dir, 'image', '%s.png' % (name)))
        img = np.array(img)
        # img = img[:,:,:3] * 255
        # img = img.astype(np.int32)
        return img

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        data_dir = self.data_dir
        name = self.filenames[image_id]
        mask = cv2.imread(os.path.join(data_dir, 'mask', '%s.png' % (name)))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # print(multi_mask.shape)
        return mask



# draw  ----------------------------------------------------------------

def color_overlay_to_mask(image):
    H,W = image.shape[:2]

    mask = np.zeros((H,W),np.int32)
    unique_color = set( tuple(v) for m in image for v in m )

    #print(len(unique_color))
    count=0
    for color in unique_color:
        #print(color)
        if color ==(0,0,0): continue

        thresh = (image==color).all(axis=2)
        label  = skimage.morphology.label(thresh)

        index = [label!=0]
        count = mask.max()
        mask[index] =  label[index]+count

    return mask


def mask_to_color_overlay(mask, image=None, color=None):

    height,width = mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()

    num_instances = int(mask.max())
    if num_instances==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_instances))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_instances) ]

    for i in range(num_instances):
        overlay[mask==i+1]=color[i]

    return overlay



def mask_to_contour_overlay(mask, image=None, color=[255,255,255]):

    height,width = mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()

    num_instances = int(mask.max())
    if num_instances==0: return overlay

    for i in range(num_instances):
        overlay[mask_to_inner_contour(mask==i+1)]=color

    return overlay

# modifier  ----------------------------------------------------------------

def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour

def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


##-------------------------------------------------------------------------
def mask_to_distance(mask):

    H,W = mask.shape[:2]
    distance = np.zeros((H,W),np.float32)

    num_instances = mask.max()
    for i in range(num_instances):
        instance = mask==i+1
        d = ndimage.distance_transform_edt(instance)
        d = d/(d.max()+0.01)
        distance = distance+d

    distance = distance.astype(np.float32)
    return distance



def mask_to_annotation(mask):
    foreground = (mask!=0).astype(np.float32)

    H,W = mask.shape[:2]
    border = np.zeros((H,W),np.float32)
    distance   = mask_to_distance(mask)
    y,x = np.where( np.logical_and(distance>0, distance<0.5) )
    border[y,x] = 1

    return  foreground, border


def instance_to_mask(instance):
    H,W = instance.shape[1:3]
    mask = np.zeros((H,W),np.int32)

    num_instances = len(instance)
    for i in range(num_instances):
         mask[instance[i]>0] = i+1

    return mask

def mask_to_instance(mask):
    H,W = mask.shape[:2]
    num_instances = mask.max()
    instance = np.zeros((num_instances,H,W), np.float32)
    for i in range(num_instances):
         instance[i] = mask==i+1

    return instance

def instance_to_multi_mask(instance):
    H,W = instance[0].shape[:2]
    multi_mask = np.zeros((H,W),np.int32)

    num_masks = len(instance)
    for i in range(num_masks):
         multi_mask[instance[i]>0] = i+1

    return multi_mask

def get_weights(mask):
    w0 = 10
    sigma = 5

    merged_mask = mask > 0
    masks = mask_to_instance(mask)

    distances = np.array([ndimage.distance_transform_edt(m == 0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)

    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)
    
#     plt.imshow(d1)
#     plt.colorbar()
#     plt.show()
    
#     plt.imshow(d2)
#     plt.colorbar()
#     plt.show()
    
    weight = w0 * np.exp(-(d1 + d2) ** 2 / (2 * sigma ** 2)).astype(np.float32)
    weight = 1 + (merged_mask == 0) * weight
    return weight

def erode_mask(mask):
    masks = mask_to_instance(mask)
    kernel = np.ones((3,3),np.uint8)
    
    if len(masks) == 0:
        print('No mask here')
        import sys
        sys.stdout.flush()

    for i in range(len(masks)):
        masks[i] = cv2.erode(masks[i], kernel, iterations = 1)

    mask = instance_to_multi_mask(masks)
    return mask


# check ##################################################################################3

def run_check_train_dataset_reader():

    dataset = ScienceDataset(
        'train1_ids_gray2_500',
        #'disk0_ids_dummy_9',
        #'merge1_1',
        mode='train',transform = None,
    )

    for n in range(len(dataset)):
        i=n #13  #=
        image, truth_mask, index = dataset[i]

        folder, name   = dataset.ids[index].split( '/')
        print('%05d %s' %(i,name))

        # image1 = random_transform(image, u=0.5, func=process_gamma, gamma=[0.8,2.5])
        # image2 = process_gamma(image, gamma=2.5)

        #image1 = random_transform(image, u=0.5, func=do_process_custom1, gamma=[0.8,2.5],alpha=[0.7,0.9],beta=[1.0,2.0])
        #image1 = random_transform(image, u=0.5, func=do_unsharp, size=[9,19], strength=[0.2,0.4],alpha=[4,6])
        #image1 = random_transform(image, u=0.5, func=do_speckle_noise, sigma=[0.1,0.5])

        #image1, truth_mask1 = random_transform2(image, truth_mask, u=0.5, func=do_shift_scale_rotate2, dx=[0,0],dy=[0,0], scale=[1/2,2], angle=[-45,45])
        #image1, truth_mask1 = random_transform2(image, truth_mask, u=0.5, func=do_elastic_transform2, grid=[16,64], distort=[0,0.5])



        # image_show('image',image,1)
        # color_overlay = mask_to_color_overlay(truth_mask)
        # image_show('color_overlay',color_overlay,1)
        #
        #
        # image_show('image1',image1,1)
        # #image_show('image2',image2,1)
        # color_overlay1 = mask_to_color_overlay(truth_mask1)
        # image_show('color_overlay1',color_overlay1,1)


        truth_foreground, truth_border = mask_to_annotation(truth_mask)

        contour_overlay = mask_to_contour_overlay(truth_mask,  image, [0,255,0])
        image_show('contour_overlay',contour_overlay,resize=1)
        image_show_norm('truth_foreground',truth_foreground,resize=1)
        image_show_norm('truth_border',truth_border,resize=1)


        cv2.waitKey(0)
        continue






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_dataset_reader()

    print( 'sucess!')












