from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from xml.dom.minidom import parse
import torch
from model import CNNEncoder
from model import FullyConnect
import numpy as np
from PIL import ImageDraw,Image
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from resnet18 import resnet18

JPEGImages = './SCUT_HEAD_Part_A/JPEGImages/'
Annotations = './SCUT_HEAD_Part_A/Annotations/'
# JPEGImages = './SCUT_HEAD/JPEGImages/'
# Annotations = './SCUT_HEAD/Annotations/'

X_SIZE = 512
Y_SIZE = 256
x_cnt = 16
y_cnt = 8
each_box = [32,16]
is_GPU = True
GPU = 2
LEARNING_RATE = 0.0008
BATCH_SIZE = 48

class SCUT_HEAD_Dataset(Dataset):

    def __init__(self, image_roots, label_root_dict, x_size, y_size,  split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.split = split
        self.image_roots = image_roots
        self.label_root_dict = label_root_dict
        self.x_size = x_size
        self.y_size = y_size

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(os.path.join(JPEGImages, image_root))
        ori_x = image.size[0]
        ori_y = image.size[1]
        image = image.convert('RGB')
        image = image.resize((self.x_size, self.y_size))
        if self.transform is not None:
            image = self.transform(image)
        label_root = self.label_root_dict[image_root.split('.')[0]]
        domTree = parse(os.path.join(Annotations,label_root))
        # 文档根元素
        rootNode = domTree.documentElement
        objects = rootNode.getElementsByTagName("object")
        coordinate_list = []
        for obj in objects:
            banbox = obj.getElementsByTagName("bndbox")[0]
            x_min = banbox.getElementsByTagName('xmin')[0].childNodes[0].data
            y_min = banbox.getElementsByTagName('ymin')[0].childNodes[0].data
            x_max = banbox.getElementsByTagName('xmax')[0].childNodes[0].data
            y_max = banbox.getElementsByTagName('ymax')[0].childNodes[0].data
            x_min = int(x_min) * self.x_size * 1.0 / ori_x
            y_min = int(y_min) * self.y_size * 1.0 / ori_y
            x_max = int(x_max) * self.x_size * 1.0 / ori_x
            y_max = int(y_max) * self.y_size * 1.0 / ori_y

            coordinate_list.append((x_min,y_min,x_max,y_max))
        return image_root, image, coordinate_list

def collate_fn(batch):
    batch = list(zip(*batch))
    sample = batch[1]
    root = batch[0]
    labels = batch[2]
    del batch
    return root, sample, labels

def align_labels(labels) :
    max_len = -1
    for label in labels:
        if(len(label) > max_len):
            max_len = len(label)

    for label in labels:
        for i in range(max_len - len(label)):
            label.append((-1))
    return labels

def compute_IOU(box_1, box_2):
    x_min_1,y_min_1,x_max_1,y_max_1 = box_1[0], box_1[1],box_1[2],box_1[3]
    x_min_2,y_min_2,x_max_2,y_max_2 = box_2[0], box_2[1],box_2[2],box_2[3]

    inter_x_min = max(x_min_1, x_min_2)
    inter_y_min = max(y_min_1, y_min_2)
    inter_x_max = min(x_max_1, x_max_2)
    inter_y_max = min(y_max_1, y_max_2)

    width = max(0, inter_x_max - inter_x_min)
    height = max(0, inter_y_max - inter_y_min)

    inter_area = width * height * 1.0

    merge_area = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * 1.0 + (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * 1.0

    IOU = inter_area*1.0 / (merge_area - inter_area + 1e-6)

    return IOU

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

def draw_rectangle_by_point(image,new_img_file_path,points, x_resize, y_resize):

    ori_img_size = image.size
    image = image.resize((x_resize, y_resize))

    draw = ImageDraw.Draw(image)

    for point in points:
        first_point = (int(point[0] * x_resize * 1.0 / ori_img_size[0]), int(point[1] * y_resize * 1.0 / ori_img_size[1]))
        last_point = (int(point[2] * x_resize * 1.0 / ori_img_size[0]), int(point[3] * y_resize * 1.0 / ori_img_size[1]))
        # print("左上角：",first_point)
        # print("右下角：",last_point)

        draw.rectangle([first_point[0], first_point[1], last_point[0], last_point[1]], outline=(255,0,0))
    image.save(new_img_file_path)



# 将1920*1080的图片划分为 48*27的格子，每个格子是40*40
def split_img_to_box(x, y, each_box, boxes):
    each_box = [32, 16]
    x = 16
    y = 8
    # torch.set_printoptions(profile="full")
    split_info = torch.zeros((y+1,x+1)) # 从 1 1 开始
    split_info_coor = {}
    # print(split_info)
    # print(split_info.shape)
    for box in boxes:
        # print(box)
        mid_coord_x = box[0]+(box[2]-box[0])*1.0/2
        mid_coord_y = box[1]+(box[3]-box[1])*1.0/2

        box_index_x = mid_coord_x // each_box[0]
        box_index_y = mid_coord_y // each_box[1]
        if(mid_coord_x % each_box[0] > 0):
            box_index_x = box_index_x + 1
        if(mid_coord_y % each_box[1] > 0):
            box_index_y = box_index_y + 1

        box_index_y = min(y, box_index_y)
        box_index_x = min(x, box_index_x)

        # print(box_index_x)
        # print(box_index_y)
        split_info[int(box_index_y)][int(box_index_x)] = 1
        if(int(box_index_y) not in split_info_coor):
            split_info_coor[int(box_index_y)] = {}
        split_info_coor[int(box_index_y)][int(box_index_x)] = box

    # print(split_info)
    # print(split_info_coor)
    return split_info,split_info_coor

def loss(split_info_list, final_outs,split_info_coor_list,input):
    # final_out[x][y]
    # split_info[y+1][x+1]
    predict_coor_tensor = None
    true_coor_tensor = None

    predict_contain_object_confidence_tensor = None
    true_contain_object_confidence_tensor = None

    predict_no_contain_object_confidence_tensor = None

    # predict_cls = None
    # true_cls = None

    predict_y_cls = None
    true_y_cls = None

    predict_n_cls = None
    true_n_cls = None

    softmax_0 = nn.Softmax(dim=0)
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    if(is_GPU):
        mse_loss = mse_loss.cuda(GPU)
        cross_entropy_loss = cross_entropy_loss.cuda(GPU)
    id = -1
    for final_out in final_outs:
        id += 1
        split_info = split_info_list[id]
        split_info_coor = split_info_coor_list[id]
        for x in range(final_out.shape[0]):
            for y in range(final_out.shape[1]):
                if(split_info[y+1][x+1] == 1): # 包含物体
                    ############ 坐标损失 + 分类损失 #####################
                    true_bounding_box = split_info_coor[y+1][x+1]
                    # img = transform_invert(input.cpu().detach(), transforms.Compose([transforms.ToTensor()]))
                    # img.save('sss.jpg')
                    # draw_rectangle_by_point(img, './fuck.jpg',( true_bounding_box), X_SIZE, Y_SIZE)
                    # print("xxxxx")
                    true_bounding_box = torch.tensor(true_bounding_box).unsqueeze(0).repeat(2,1)
                    if (predict_contain_object_confidence_tensor == None):
                        predict_contain_object_confidence_tensor = final_out[x][y][4].unsqueeze(0)
                        predict_contain_object_confidence_tensor = torch.cat(
                            (predict_contain_object_confidence_tensor, final_out[x][y][9].unsqueeze(0)), 0)
                    else:
                        predict_contain_object_confidence_tensor = torch.cat((predict_contain_object_confidence_tensor,
                                                                              final_out[x][y][4].unsqueeze(0),
                                                                              final_out[x][y][9].unsqueeze(0)), 0)

                    predict_bounding_box_1 = (X_SIZE*1.0*final_out[x][y][0],
                                              Y_SIZE*1.0*final_out[x][y][1],
                                              X_SIZE*1.0*final_out[x][y][2],
                                              Y_SIZE*1.0*final_out[x][y][3])
                    predict_bounding_box_2 = (X_SIZE*1.0*final_out[x][y][5],
                                              Y_SIZE*1.0*final_out[x][y][6],
                                              X_SIZE*1.0*final_out[x][y][7],
                                              Y_SIZE*1.0*final_out[x][y][8])

                    predict_bounding_box = torch.cat((X_SIZE*1.0*final_out[x][y][0].unsqueeze(0),
                                                      Y_SIZE*1.0*final_out[x][y][1].unsqueeze(0),
                                                      X_SIZE*1.0*final_out[x][y][2].unsqueeze(0),
                                                      Y_SIZE*1.0*final_out[x][y][3].unsqueeze(0),
                                                      X_SIZE * 1.0 * final_out[x][y][5].unsqueeze(0),
                                                      Y_SIZE * 1.0 * final_out[x][y][6].unsqueeze(0),
                                                      X_SIZE * 1.0 * final_out[x][y][7].unsqueeze(0),
                                                      Y_SIZE * 1.0 * final_out[x][y][8].unsqueeze(0)
                                                      ), 0)

                    if(true_coor_tensor == None):
                        true_coor_tensor = true_bounding_box
                    else:
                        true_coor_tensor = torch.cat((true_coor_tensor,true_bounding_box), 0)
                    if(predict_coor_tensor == None):
                        predict_coor_tensor = predict_bounding_box
                    else:
                        predict_coor_tensor = torch.cat((predict_coor_tensor,predict_bounding_box), 0)

                    predict = softmax_0(final_out[x][y][10:12]) # 是人头  和 不是人头 的概率
                    if(predict_y_cls == None):
                        predict_y_cls = predict.unsqueeze(0)
                    else:
                        predict_y_cls = torch.cat((predict_y_cls, predict.unsqueeze(0)), 0)

                    temp = torch.tensor([1.0, 0.0]).unsqueeze(0)
                    if(true_y_cls == None):
                        true_y_cls = temp
                    else:
                        true_y_cls = torch.cat((true_y_cls, temp), 0)


                    ############ 包含人头的置信度损失  #####################
                    for pbd in [predict_bounding_box_1, predict_bounding_box_2]:
                        if(pbd[2].item() <= pbd[0].item() or
                        pbd[3].item() <= pbd[1].item()):
                            iou = 0.0
                        else:
                            iou = compute_IOU((pbd[0].item(),pbd[1].item(),pbd[2].item(),pbd[3].item()), split_info_coor[y+1][x+1])

                        iou = torch.tensor(iou).unsqueeze(0)
                        if(is_GPU):
                            iou = iou.cuda(GPU)

                        if(true_contain_object_confidence_tensor == None):
                            true_contain_object_confidence_tensor = iou
                        else:
                            true_contain_object_confidence_tensor = torch.cat((true_contain_object_confidence_tensor, iou), 0)


                ############ 不包含人头的置信度损失  #####################
                elif (split_info[y + 1][x + 1] == 0):  # 不包含物体
                    if (predict_no_contain_object_confidence_tensor == None):
                        predict_no_contain_object_confidence_tensor = final_out[x][y][4].unsqueeze(0)
                        predict_no_contain_object_confidence_tensor = torch.cat(
                            (predict_no_contain_object_confidence_tensor, final_out[x][y][9].unsqueeze(0)), 0)
                    else:
                        predict_no_contain_object_confidence_tensor = torch.cat((predict_no_contain_object_confidence_tensor,
                                                                                 final_out[x][y][4].unsqueeze(0),
                                                                                 final_out[x][y][9].unsqueeze(0)), 0)

                    predict = softmax_0(final_out[x][y][10:12])  # 是人头  和 不是人头 的概率
                    if (predict_n_cls == None):
                        predict_n_cls = predict.unsqueeze(0)
                    else:
                        predict_n_cls = torch.cat((predict_n_cls, predict.unsqueeze(0)), 0)

                    temp = torch.tensor([ 0.0, 1.0]).unsqueeze(0)
                    if (true_n_cls == None):
                        true_n_cls = temp
                    else:
                        true_n_cls = torch.cat((true_n_cls, temp), 0)

    true_no_contain_object_confidence_tensor = torch.zeros(predict_no_contain_object_confidence_tensor.shape)
    # true_cls = torch.ones(predict_cls.shape)

    # print("predict_coor_tensor:", predict_coor_tensor)
    # print("true_coor_tensor:", true_coor_tensor)
    # print("true_coor_tensor1:", true_coor_tensor.flatten())
    #
    # print("predict_contain_object_confidence_tensor:", predict_contain_object_confidence_tensor)
    # print("true_contain_object_confidence_tensor:",true_contain_object_confidence_tensor)
    #
    # print("predict_no_contain_object_confidence_tensor:", predict_no_contain_object_confidence_tensor)
    # print("true_no_contain_object_confidence_tensor", true_no_contain_object_confidence_tensor)
    #
    # print("true_n_cls:", true_n_cls)
    # print("predict_n_cls:",predict_n_cls)
    # print("true_y_cls:", true_y_cls)
    # print("predict_y_cls:",predict_y_cls)
    # print("true_cls:", true_n_cls.shape)
    # print("predict_cls:",predict_n_cls.shape)
    # print("true_cls:", true_y_cls.shape)
    # print("predict_cls:",predict_y_cls.shape)
    # print('ssss')

    if(true_coor_tensor == None or
            predict_coor_tensor == None or
            predict_contain_object_confidence_tensor == None or
            true_contain_object_confidence_tensor == None or
            predict_no_contain_object_confidence_tensor == None or
            true_no_contain_object_confidence_tensor == None or
            predict_y_cls == None or
            true_y_cls == None or
            predict_n_cls == None or
            true_n_cls == None):
        return -1,-1,-1,-1,-1

    true_coor_tensor = true_coor_tensor.flatten()
    if(is_GPU):
        true_coor_tensor = Variable(true_coor_tensor).cuda(GPU)
        true_contain_object_confidence_tensor = Variable(true_contain_object_confidence_tensor).cuda(GPU)
        true_no_contain_object_confidence_tensor = Variable(true_no_contain_object_confidence_tensor).cuda(GPU)
        true_y_cls = Variable(true_y_cls).cuda(GPU)
        true_n_cls = Variable(true_n_cls).cuda(GPU)


    coor_mse_loss = mse_loss(predict_coor_tensor, true_coor_tensor)
    confidence_mse_loss_N = mse_loss(predict_contain_object_confidence_tensor, true_contain_object_confidence_tensor)
    confidence_mse_loss_Y = mse_loss(predict_no_contain_object_confidence_tensor, true_no_contain_object_confidence_tensor)
    y_cls_mse_loss = mse_loss(predict_y_cls, true_y_cls.squeeze(1))
    n_cls_mse_loss = mse_loss(predict_n_cls, true_n_cls.squeeze(1))

    total_loss = 5*coor_mse_loss + confidence_mse_loss_Y + 0.2*confidence_mse_loss_N + y_cls_mse_loss + n_cls_mse_loss

    # print(total_loss)
    # return total_loss
    return coor_mse_loss, confidence_mse_loss_Y, confidence_mse_loss_N, y_cls_mse_loss, n_cls_mse_loss


def test():
    resnet18_model = resnet18()
    fc = FullyConnect()

    if (is_GPU):
        fc = fc.cuda(GPU)
        resnet18_model = resnet18_model.cuda(GPU)
    total_loss = 32.81605911254883
    # 217.16387939453125
    # 259.6724853515625
    if os.path.exists(str("./models/" + str(total_loss)+ "_res.pkl")):
        resnet18_model.load_state_dict(torch.load(str("./models/" + str(total_loss)+ "_res.pkl")))
        print("load  resnet18_model success")
    if os.path.exists(str("./models/" + str(total_loss)+ "_fc.pkl")):
        fc.load_state_dict(torch.load(str("./models/" + str(total_loss)+ "_fc.pkl")))
        print("load  fc success")

    # root = './cs'
    image_root = 'PartA_00000.jpg'
    # image_root = '1.jpg'
    #
    image = Image.open(os.path.join(JPEGImages, image_root))
    # image = Image.open(os.path.join(root, image_root))

    image = image.convert('RGB')
    image = image.resize((X_SIZE, Y_SIZE))
    transform = transforms.Compose([transforms.ToTensor()])
    if transform is not None:
        input = transform(image)

    resnet18_model.eval()
    fc.eval()
    softmax_0 = nn.Softmax(dim=0)
    with torch.no_grad():
        out = fc(resnet18_model(Variable(input).unsqueeze(0).cuda(GPU)))
        out = out.reshape(-1, x_cnt, y_cnt, (2 * 5 + 2))
        out = out[0]

        coor_list = []
        bb_list = []
        for x in range(x_cnt):
            for y in range(y_cnt):
                # 每个 grid cell 有两个 boundingbox
                conf1 = out[x][y][4] # 第一个 bb 的 置信度
                conf2 = out[x][y][9] # 第二个 bb的置信度
                predict_cls = softmax_0(out[x][y][10:12])

                bb1 = torch.mul(conf1, predict_cls)
                coor_list.append((X_SIZE*1.0*out[x][y][0],Y_SIZE*1.0*out[x][y][1],X_SIZE*1.0*out[x][y][2],Y_SIZE*1.0*out[x][y][3]))
                bb_list.append(bb1)
                bb2 = torch.mul(conf2, predict_cls)
                coor_list.append((X_SIZE*1.0*out[x][y][5],Y_SIZE*1.0*out[x][y][6],X_SIZE*1.0*out[x][y][7],Y_SIZE*1.0*out[x][y][8]))
                bb_list.append(bb2)

        print("bb_list:", bb_list)
        # print("coor_list", coor_list)
        # print("fuck")

        thresh_1 = 0.2
        thresh_2 = 0.5
        for i in range(bb_list[0].shape[0]):
            row_dict = {}
            for j in range(len(bb_list)):
                if(bb_list[j][i] < thresh_1 or (coor_list[j][0] >= coor_list[j][2] or coor_list[j][1] >= coor_list[j][3])):
                    row_dict[j] = 0
                else:
                    row_dict[j] = bb_list[j][i]
            # print("row_dict:", row_dict)
            row_dict_tuple = sorted(row_dict.items(), key=lambda row_dict: row_dict[1], reverse=True) # 从高到低排序 0为索引  1为得分
            row_dict_tuple = [list(x) for x in row_dict_tuple]
            max = row_dict_tuple[0]

            for k in range(1, len(row_dict_tuple)):
                if(row_dict_tuple[k][1] == 0):
                    continue

                iou = compute_IOU(coor_list[max[0]], coor_list[row_dict_tuple[k][0]])
                if(iou >= thresh_2 ):
                    row_dict_tuple[k][1] = 0

            # 更新 bblist
            for row_info in row_dict_tuple:
                bb_list[row_info[0]][i] = row_info[1]
            # print("row_dict_tuple:", row_dict_tuple)
            # print("update bb_list:", bb_list)
            # print("dddddd")
    draw_rectangle_list = []
    index = 0
    for bb in bb_list:
        cls = torch.argmax(bb) # cls 0/1 0人头 1不是人头
        score = torch.max(bb)
        if(score > 0 and cls == 0):
            print("bb:", bb)
            print("cls:", cls)
            print("coor:", coor_list[index])
            draw_rectangle_list.append(coor_list[index])

        index += 1
    print("update_bb_list:", bb_list)

    img = transform_invert(input, transforms.Compose([transforms.ToTensor()]))
    # img.save('sss.jpg')
    draw_rectangle_by_point(img, './fuck.jpg', draw_rectangle_list, X_SIZE, Y_SIZE)
    # print('test done')

#

#
if __name__ == '__main__':
    image_root = os.listdir(JPEGImages)
    label_root = os.listdir(Annotations)
    label_root_dict = {}
    for label in label_root:
        label_root_dict[label.split('.')[0]] = label

    # print("image_root:", image_root)
    # print("label_root_dict:", label_root_dict)
    # iou = compute_IOU([0, 0, 2, 2], [2, 2, 4, 4])
    # print(iou)

    dataset = SCUT_HEAD_Dataset(image_root, label_root_dict, X_SIZE, Y_SIZE, transform=transforms.Compose([transforms.ToTensor()]))

    resnet18_model = resnet18(pretrained=False)
    fc = FullyConnect()

    if(is_GPU):
        fc = fc.cuda(GPU)
        resnet18_model = resnet18_model.cuda(GPU)

    fc_optim = torch.optim.Adam(fc.parameters(),lr=LEARNING_RATE)
    fc_scheduler = StepLR(fc_optim,step_size=99999999,gamma=0.5)

    resnet18_model_optim = torch.optim.Adam(resnet18_model.parameters(),lr=LEARNING_RATE)
    resnet18_model_scheduler = StepLR(resnet18_model_optim,step_size=99999999,gamma=0.5)

    resnet18_model.train()
    fc.train()

    for epoch in range(10000):
        loss_total = 0
        cnt_num = 0
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        for data in data_loader:
            train_data = torch.stack(list(data[1]), dim=0)
            cnt_num += 1
            labels = data[2] # batch_size个样本的所有框
            image_root = data[0]
            # print(train_data.shape)
            # img = transform_invert(train_data[0], transforms.Compose([transforms.ToTensor()]))
            # # img.save('rnm.jpg')
            # print(labels[0]) # 第一个样本的所有框
            # draw_rectangle_by_point(img, image_root[0], labels[0] , X_SIZE, Y_SIZE)

            if(is_GPU):
                train_data = train_data.cuda(GPU)

            split_info_list = []
            split_info_coor_list = []
            for id in range(len(train_data)):
                split_info,split_info_coor = split_img_to_box(x_cnt, y_cnt, each_box, labels[id])
                split_info_list.append(split_info)
                split_info_coor_list.append(split_info_coor)
                id += 1

            res18_out = resnet18_model(train_data)
            fc_out = fc(res18_out)

            final_outs = fc_out.reshape(-1, x_cnt, y_cnt, (2*5+2))
            coor_mse_loss, confidence_mse_loss_Y, confidence_mse_loss_N, y_cls_mse_loss, n_cls_mse_loss = loss(split_info_list, final_outs, split_info_coor_list, train_data)
            if(coor_mse_loss == -1):
                continue
            total_loss = 5 * coor_mse_loss + confidence_mse_loss_Y + 0.5 * confidence_mse_loss_N + y_cls_mse_loss +  0.5 *n_cls_mse_loss


            loss_total += total_loss.item()

            resnet18_model.zero_grad()
            fc.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(resnet18_model.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(fc.parameters(), 0.5)

            resnet18_model_optim.step()
            fc_optim.step()
            print("handle:", str(cnt_num) + "/" + str(len(data_loader)), "-->", "coor_mse_loss:",
                  coor_mse_loss.item(), "confidence_mse_loss_Y:", confidence_mse_loss_Y.item(),
                  "confidence_mse_loss_N:", confidence_mse_loss_N.item(), "y_cls_mse_loss:", y_cls_mse_loss.item(),"n_cls_mse_loss:", n_cls_mse_loss.item(),
                  "total_loss", total_loss.item())

            # if(cnt % 100 == 0):
            #     torch.save(resnet18_model.state_dict(), "./models1/" + str(total_loss.item()) + "_res.pkl")
            #     torch.save(fc.state_dict(), "./models1/" + str(total_loss.item()) + "_fc.pkl")
            #     print("save models")

        print("epoch : ===== :", epoch, "loss_total:", loss_total)

        if(epoch % 3 == 0):
            torch.save(resnet18_model.state_dict(), "./models/" + str(total_loss.item())+ "_res.pkl")
            torch.save(fc.state_dict(), "./models/" + str(total_loss.item())+ "_fc.pkl")
            print("save models")
#
#
#
