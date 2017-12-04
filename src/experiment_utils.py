def get_ground_truth(filepath, img_filepath, num):
    gt = sio.loadmat(filepath)
    img = get_image(img_filepath)
    image_info = gt['image_info']
    value = image_info[0,0]
    assert len(value['location']) == 1
    ground = None
    for i in value['location']:
        assert len(i) == 1
        for j in i:
            ground = j 
    #print ground
    d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
    d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)

    d_map = create_dotmaps(ground / 4.0, d_map_h, d_map_w)
    #patches = prep_frame(ground)
    #for p in patches:
        #print patches.sum()

    p_h = int(math.floor(float(img.shape[0]) / 3.0))
    p_w = int(math.floor(float(img.shape[1]) / 3.0))
    d_map_ph = int(math.floor(math.floor(p_h / 2.0) / 2.0))
    d_map_pw = int(math.floor(math.floor(p_w / 2.0) / 2.0))
    
    py = 0
    py2 = 0
    count = 1

    finals = []

    for j in range(3):
        px = 0
        px2 = 0
        for k in range(3):
            final_image = img[py:py + p_h, px: px + p_w, :]
            final_gt = d_map[py2: py2 + d_map_ph, px2: px2 + d_map_pw]                 
            px = px + p_w 
            px2 = px2 + d_map_pw
            #self.save_image(final_image, i, count)
            #self.save_gt(final_gt, i, count)
            finals.append(final_gt.sum())
            count += 1
        py = py + p_h
        py2 = py2 + d_map_ph 

    print num
    print finals
    print sum(finals)
    print

def create_dotmaps(gt, img_h, img_w):
    d_map = np.zeros((int(img_h), int(img_w)))

    gt = gt[gt[:, 0] < img_w, :]
    gt = gt[gt[:, 1] < img_h, :]

    for i in range(gt.shape[0]):
        x = int(max(1, math.floor(gt[i, 0]))) - 1
        y = int(max(1, math.floor(gt[i, 1]))) - 1
        d_map[y, x] = 1.0
    return d_map

def get_image(img_filepath):
    img = pli.open(img_filepath)
    img = np.asarray(img, dtype=np.uint8)
    return img
