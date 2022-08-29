import os, datetime
import argparse
import platform
import numpy as np
import matplotlib.pyplot as plt
import cv2
import kornia as kn

import torch
from torch.utils.data import ConcatDataset, DataLoader

from model import DexiNed
from ocr_dataset import OCRSyntheticDataset, OCRTrainDataset, OCRValidDataset
from datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from transform.data_preprocessing import TrainAugmentation_Synth, TestTransform, TrainAugmentation_normalOCR
from losses import *
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result, count_parameters)

def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device, training_dir, args):

    model.train()
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]  # New BDCN  loss
    loss_avg = []

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)
        labels = sample_batched['labels'].to(device)

        preds_list = model(images)

        loss_list = []
        for preds, l_w in zip(preds_list, l_weight):
            loss_list.append(criterion(preds, labels, l_w))

        loss = sum(loss_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())

        if batch_id % 5 == 0:
            print(f'Epoch: {epoch} batch {batch_id}/{len(dataloader)} Loss : {loss.item()}')

        # 배치마다 각 depth에서(논문에서 말한 7계층) 나온 Edge Map을 print
        if batch_id % args.log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[0])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[0])

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[0]
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1] * 0.8), int(vis_imgs.shape[0] * 0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), loss.item())

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(training_dir, str(epoch), f'[BN{batch_id}][T{len(dataloader)}]_results.png'), vis_imgs)
        ###########################################################

    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def validate_one_epoch(epoch, dataloder, model, device, output_dir, args=None):
    model.eval()

    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloder):
            images = sample_batched["images"].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape'] # NCHW
            folder_names = sample_batched['folder_name']
            preds = model(images)

            # save Image
            valid_save_dir = os.path.join(output_dir, str(epoch), "valid_result")
            os.makedirs(valid_save_dir, exist_ok=True)
            for tensor_img, file_name, img_shape, folder_name in zip(preds[-1], file_names, image_shape, folder_names):
                os.makedirs(os.path.join(valid_save_dir, folder_name), exist_ok=True)
                img_shape = np.array(img_shape)

                img_vis = kn.utils.tensor_to_image(torch.sigmoid(tensor_img))
                img_vis = (255.0 *(1.0 - img_vis)).astype(np.uint8)
                img_vis = cv2.resize(img_vis, dsize=(img_shape[1], img_shape[0]))
                cv2.imwrite(os.path.join(valid_save_dir, folder_name, file_name), img_vis)
            #######


def test():
    pass

def parse_args():
    parser = argparse.ArgumentParser(description= "my Dexined Trainer")

    parser.add_argument('--device', type=int, default=0, help='select gpu device')
    parser.add_argument('--output_dir', type=str, help='train or test output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adjust_lr', type=int, default=[10, 15])

    parser.add_argument('--mode', type=str, default='test', help='mode select(train or test)')

    parser.add_argument('--model_path', type=str, default=None, help='input model path')

    parser.add_argument('--trainset_list', nargs="+", default=None, type=str, help='input train data path')
    parser.add_argument('--testset_list', nargs="+", default=None, type=str, help='input test data path')

    parser.add_argument('--mean_pixel_values',
                        default=[103.939, 116.779, 123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=50,
                        help='The number of batches to wait before printing test predictions.')

    IS_LINUX = True if platform.system() == "Linux" else False
    TRAIN_DATA = DATASET_NAMES[0]  # BIPED=0, MDBD=6
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    parser.add_argument('--train_list',
                        type=str,
                        default=train_inf['train_list'],
                        help='Dataset sample indices list.')
    TEST_DATA = DATASET_NAMES[0]  # [parser.parse_args().choose_test_data] # max 8
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir']
    parser.add_argument('--test_list',
                        type=str,
                        default=test_inf['test_list'],
                        help='Dataset sample indices list.')

    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')

    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)

    parser.add_argument('--test_img_width',
                        type=int,
                        default=test_inf["img_width"],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=test_inf["img_height"],
                        help='Image height for testing.')
    args = parser.parse_args()

    return args

def main(args):

    init_epoch = 0

    date_str = datetime.datetime.today().strftime("%Y_%m_%d_%H%M%S")
    training_dir = os.path.join(args.output_dir, date_str)
    os.makedirs(training_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    model = DexiNed().to(device)

    if args.mode == 'train':
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Training restarted from> {args.model_path}")

            model_name = args.model_path.split('/')[-1]
            if '_model.pth' in model_name:
                init_epoch = int(model_name.replace('_model.pth', '')) + 1

        train_datasets = []
        for data_path in args.trainset_list:
            if 'BIPED' in data_path:
                train_dataset = BipedDataset(data_path,
                                             img_width=352,
                                             img_height=352,
                                             mean_bgr=args.mean_pixel_values[0:3] if len(
                                             args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                             train_mode='train',
                                             arg=args)

            elif 'Synth' in data_path:
                train_transform = TrainAugmentation_Synth(size=352,
                                                          mean=np.array([0.485, 0.456, 0.406]),
                                                          std=np.array([0.229, 0.224, 0.225]))
                train_dataset = OCRSyntheticDataset(data_path,
                                                    transform=train_transform,
                                                    target_transform=None,
                                                    is_test=False,
                                                    image_size=352)
            else:
                train_transform = TrainAugmentation_normalOCR(size=352,
                                                              mean=np.array([0.485, 0.456, 0.406]),
                                                              std=np.array([0.229, 0.224, 0.225]))
                train_dataset = OCRTrainDataset(data_path,
                                                transform=train_transform,
                                                target_transform=None,
                                                is_test=False,
                                                image_size=352)

            train_datasets.append(train_dataset)

        train_datasets = ConcatDataset(train_datasets)

        valid_datasets = []
        for data_path in args.testset_list:
            if 'BIPED' in data_path:
                valid_dataset = TestDataset(data_path,
                                            test_data=DATASET_NAMES[0],
                                            img_width=352,
                                            img_height=352,
                                            mean_bgr=args.mean_pixel_values[0:3] if len(
                                                args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                            test_list=args.test_list,
                                            arg=args
                                            )
            elif 'Synth' in data_path:
                pass
            else:
                test_transform = TestTransform(size=352,
                                              mean=np.array([0.485, 0.456, 0.406]),
                                              std=np.array([0.229, 0.224, 0.225]))
                valid_dataset = OCRValidDataset(root=data_path,
                                                transform=test_transform,
                                                target_transform=None,
                                                is_test=False)

            valid_datasets.append(valid_dataset)

        valid_datasets = ConcatDataset(valid_datasets)

        dataloader_train = DataLoader(train_datasets,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)

        dataloader_valid = DataLoader(valid_datasets,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers)

    elif args.mode == 'test':
        valid_datasets = []
        for data_path in args.testset_list:
            if 'BIPED' in data_path:
                valid_dataset = TestDataset(data_path,
                                            test_data=DATASET_NAMES[0],
                                            img_width=352,
                                            img_height=352,
                                            mean_bgr=args.mean_pixel_values[0:3] if len(
                                                args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                            test_list=args.test_list,
                                            arg=args
                                            )
            elif 'Synth' in data_path:
                pass
            else:
                test_transform = TestTransform(size=352,
                                               mean=np.array([0.485, 0.456, 0.406]),
                                               std=np.array([0.229, 0.224, 0.225]))
                valid_dataset = OCRValidDataset(root=data_path,
                                                transform=test_transform,
                                                target_transform=None,
                                                is_test=False)

            valid_datasets.append(valid_dataset)

        valid_datasets = ConcatDataset(valid_datasets)
        dataloader_valid = DataLoader(valid_datasets,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers)

    if args.mode == 'test':
        date_str = datetime.datetime.today().strftime("%Y_%m_%d_%H%M")
        output_dir = os.path.join(args.output_dir, f'test_res_{date_str}')
        os.makedirs(output_dir, exist_ok=True)

        test(args.model_path, dataloader_valid, model, output_dir, args)

        print('---------------------------------------------------------')
        print('end of test')
        return

    criterion = bdcn_loss2

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    seed = 1022
    adjust_lr = args.adjust_lr
    lr2 = args.lr

    epoch_list = []
    loss_list = []
    for epoch in range(init_epoch, init_epoch + args.epochs):
        if epoch % 7 == 0:
            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('Random Seed Reset')

        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = lr2*0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2

        output_dir_epoch = os.path.join(training_dir, str(epoch))
        os.makedirs(output_dir_epoch, exist_ok=True)

        avg_loss = train_one_epoch(epoch,
                                   dataloader_train,
                                   model,
                                   criterion,
                                   optimizer,
                                   device,
                                   training_dir,
                                   args)

        validate_one_epoch(epoch, dataloader_valid, model, device, training_dir, args)

        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, f'{epoch}_model.pth'))

        print(f'epoch {epoch + 1} Avg Loss : {avg_loss}')
        epoch_list.append(str(epoch+1))
        loss_list.append(avg_loss)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(epoch_list, loss_list)
        plt.savefig(os.path.join(training_dir, 'loss_graph.png'))

if __name__ == "__main__":
    args = parse_args()

    print(args.train_list)
    main(args)
