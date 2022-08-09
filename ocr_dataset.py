import numpy as np
import logging
from utils.LMK_Decoder import lmk_decoder
import cv2
import os
import pathlib
import numpy as np
import math
#
# np.random.seed(0)
'''
    배경 이미지 Image_0.jpg 에서 box idx 4번째 배경을 추출해서
    Font에서 N개의 폰트를 가져와서 합성
    배경이미지는 다안써도 상관X
    Font 이미지는 클래스 별로 균형을 맞춰서..
    합성 이미지는 5만장으로 고정( Seed 를 줘서 고정? )
'''
class OCRSyntheticDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, image_size=300):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        # font ids
        self.font_ids = OCRSyntheticDataset._read_image_ids(self.root / "Font")
        self.font_idx = {_key:0 for _key in self.font_ids.keys()}
        # background ids
        self.background_ids = OCRSyntheticDataset._read_image_ids(self.root / f"Background")
        # self.background_idx = {_key: 0 for _key in self.background_ids.keys()}

        # if the labels file exists, read in the class names
        label_file_name = self.root / f"labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default classes.")
            self.class_names = ('BACKGROUND',
                                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                'U', 'V', 'W', 'X', 'Y', 'Z')


        self.class_dict = {class_name : i for i , class_name in enumerate(self.class_names)}


    def __getitem__(self, index):
        image, boxes, labels, debug_image, edge_labels = self._get_synth_image(index)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes.astype(np.float32), labels)
        edge_labels = edge_labels.transpose(2, 0, 1)
        edge_labels = edge_labels / 255.0
        return image, 1, 1, edge_labels.astype(np.float32)

    def __len__(self):
        return 30000

    def _get_synth_image(self,index):
        number_of_char = np.random.randint(2, 11) #합성 이미지에 사용할 문자의 갯수 , 한이미지에 포함되는 문자의 수는 2개 ~ 10개
        char_label = np.random.randint(1, 36, number_of_char) # 합성 이미지에 사용할 문자의 class를 정함
        '''
            합성이미지에 사용될 문자의 종류를 랜덤으로 지정함
            ex: 
                [ 1 4 3 5 4 3 4 5 ]
            0~9 , A~Z : 숫자 10개, 영대문자 26개 총 36개 문자
            label 0 : background
            label 1 : "0" ... 
        '''

        '''
            랜덤하게 정해진 문자를 실제 이미지 데이터에서 추출하는 과정
            font_ids는 클래스 별 폰트이미지의 번호를 저장하고있는 dictionary
            font_idx는 클래스 별로 뽑힌 index를 저장
        '''
        images = []
        for _char in char_label:

            image_id = self.font_ids[str(_char)][self._get_font_idx(_char)]
            image, label = self._read_image(image_id,_char)
            boxes, _ = self._get_annotation(image_id,_char,'Font')

            images.append([image,label,boxes])


        #랜덤하게 배경 이미지를 추출
        
        # 배경이미지의 높이나 넓이가 0일 경우 다시 추출
        while True:
            bg_image = self._get_background(index)
            if bg_image.shape[0] != 0 and bg_image.shape[1] != 0:
                break

        #합성 이미지 생성
        synth_image, synth_boxes, _debug_image, synth_label = self._make_synth_image(images,bg_image)

        return synth_image, synth_boxes, np.array(char_label,dtype=np.float32), _debug_image, synth_label

    def _get_font_idx(self,font):
        idx = self.font_idx[str(font)]

        self.font_idx[str(font)] += 1
        self.font_idx[str(font)] = self.font_idx[str(font)] % len(self.font_ids[str(font)])
        return idx

    def _make_synth_image(self, images, bg_image):
        # images : selected font (128 x 128)
        # bg_image : ?? x 300
        height = bg_image.shape[0]
        # width = bg_image.shape[1]
        # ratio = height / 300

        bg_image = cv2.resize(bg_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        N_chunk = np.random.randint(1, len(images))  # 1부터 3까지
        choice = np.sort(np.random.choice(len(images) - 1, N_chunk - 1, replace=False))
        choice = np.append(choice, [len(images) - 1])
        # 각 청크는 적어도 하나 이상의 원소를 포함.
        # [ 0 1 2 3 | 4 5 6| 7 8 | 9]
        chunk_list = []

        chunk_start_index = 0
        for i in range(N_chunk):
            chunk_list.append(images[chunk_start_index:choice[i] + 1])
            chunk_start_index = choice[i] + 1

        '''
            합성 폰트 chunk 생성 
            폰트간 간격 랜덤
        '''
        synth_image_list = []
        # synth_label_list = []
        synth_box_list = []
        for chunk in chunk_list:
            chunk_box_list = []
            synth_image, synth_label_img, synth_box = chunk[0][:]
            [left, top, right, bottom] = synth_box[0][:]

            ##
            # synth_label_img = cv2.bitwise_not(synth_image)
            # contour, hierachy = cv2.findContours(synth_label_img[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # synth_label_img = cv2.drawContours(np.zeros_like(cv2.bitwise_not(synth_image)), contour, -1, (255, 255, 255), 1)


            width = right - left
            random_width = np.random.randint(width + 10, 129)  # low <= X < high / half open
            left_padding = int((random_width - width) / 2)
            right_padding = left_padding

            temp_left = left
            left = left - left_padding
            if left < 0:
                left_padding = temp_left
                left = 0

            temp_right = right
            right = right + right_padding
            if right > 128:
                right_padding = 128 - temp_right
                right = 128

            left = int(round(left))
            bottom = int(round(bottom)) + 1
            right = int(round(right))
            top = int(round(top))

            synth_roi = synth_image[top:bottom, left:right]
            # synth_label_roi = synth_label_img[top:bottom, left:right]
            chunk_box_list.append([left_padding, 0, (right - left) - right_padding, (bottom - top)])  # 변화된 bbox의 위치를 갱신

            for i in range(1, len(chunk)):
                image, label_img, box = chunk[i][:]
                left, top, right, bottom = box[0][:]

                # label_img = cv2.bitwise_not(image)
                # contour, hierachy = cv2.findContours(label_img[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # label_img = cv2.drawContours(np.zeros_like(cv2.bitwise_not(image)), contour, -1,
                #                                    (255, 255, 255), 1)

                temp_left = left
                left = left - left_padding
                if left < 0:
                    left_padding = temp_left
                    left = 0

                temp_right = right
                right = right + right_padding
                if right > 128:
                    right_padding = 128 - temp_right
                    right = 128

                left = int(round(left))
                bottom = int(round(bottom)) + 1
                right = int(round(right))
                top = int(round(top))
                height = bottom - top
                _roi = image[top:bottom, left:right]
                # _label_roi = label_img[top:bottom, left:right]

                prev_roi_width = synth_roi.shape[1]
                if synth_roi.shape[0] != _roi.shape[0]:
                    _roi = cv2.resize(_roi, (_roi.shape[1], synth_roi.shape[0]), cv2.INTER_NEAREST)
                    height = synth_roi.shape[0]

                synth_roi = cv2.hconcat((synth_roi, _roi))
                # synth_label_roi = cv2.hconcat((synth_label_roi,_label_roi))
                chunk_box_list.append(
                    [prev_roi_width + left_padding, 0, prev_roi_width + (right - left) - right_padding, height])

            height, width, _ = synth_roi.shape
            aspect_ratio = np.random.random() + 1.0
            synth_roi = cv2.resize(synth_roi, dsize=(0, 0), fx=aspect_ratio, fy=aspect_ratio,
                                   interpolation=cv2.INTER_NEAREST)
            # synth_label_roi = cv2.resize(synth_label_roi,dsize=(0,0),fx=aspect_ratio,fy=aspect_ratio,interpolation=cv2.INTER_NEAREST)

            chunk_box_list = np.array(chunk_box_list) * aspect_ratio  # elemental wise multiplication

            synth_image_list.append(synth_roi)
            # synth_label_list.append(synth_label_roi)
            synth_box_list.append(chunk_box_list)

        # 청크 이미지들을 하나의 window image로 생성
        _synth_window_image = synth_image_list[0]
        # _synth_window_label = synth_label_list[0]
        _synth_window_box = synth_box_list[0]

        for i in range(1, len(synth_image_list)):
            _image = synth_image_list[i]
            # _label = synth_label_list[i]
            _box = synth_box_list[i]

            # 첫번째는 hconcat으로 가로로 붙임 , 그다음 번갈아 가면서 vconcat
            if (i % 2) == 1:
                _height_1 = _synth_window_image.shape[0]
                _height_2 = _image.shape[0]
                _width_1 = _synth_window_image.shape[1]
                _width_2 = _image.shape[1]
                if _height_1 > _height_2:
                    # concat image
                    _diff_height = _height_1 - _height_2
                    _top_padding = np.random.randint(0, _diff_height + 1)
                    _bottom_padding = _diff_height - _top_padding
                    _left_padding = np.random.randint(30, 100)
                    _image = cv2.copyMakeBorder(_image, top=_top_padding, bottom=_bottom_padding, left=_left_padding,
                                                right=0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    _synth_window_image = cv2.hconcat((_synth_window_image, _image))  # 이미지에 다음 이미지를 가로로 붙임.

                    # # concat label
                    # _label = cv2.copyMakeBorder(_label, top=_top_padding, bottom=_bottom_padding, left=_left_padding,
                    #                              right=0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    # _synth_window_label = cv2.hconcat((_synth_window_label, _label))

                    # concat box
                    _box = _box[:] + [_width_1 + _left_padding, _top_padding, _width_1 + _left_padding, _top_padding]
                    _synth_window_box = np.concatenate((_synth_window_box, _box), 0)
                else:
                    # concat image
                    _diff_height = _height_2 - _height_1
                    _top_padding = np.random.randint(0, _diff_height + 1)
                    _bottom_padding = _diff_height - _top_padding
                    _synth_window_image = cv2.copyMakeBorder(_synth_window_image, top=_top_padding,
                                                             bottom=_bottom_padding, left=0, right=0,
                                                             borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

                    _left_padding = np.random.randint(30, 100)
                    _image = cv2.copyMakeBorder(_image, top=0, bottom=0, left=_left_padding,
                                                right=0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

                    _synth_window_image = cv2.hconcat((_synth_window_image, _image))

                    # # concat label
                    # _synth_window_label = cv2.copyMakeBorder(_synth_window_label, top=_top_padding, bottom=_bottom_padding,
                    #                                          left=0, right=0,
                    #                                          borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    # _label = cv2.copyMakeBorder(_label, top=0, bottom=0, left=_left_padding,
                    #                             right=0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    # _synth_window_label = cv2.hconcat((_synth_window_label, _label))

                    # concat box
                    _synth_window_box = _synth_window_box[:] + [0, _top_padding, 0, _top_padding]
                    _box = _box[:] + [_width_1 + _left_padding, 0, _width_1 + _left_padding, 0]
                    _synth_window_box = np.concatenate((_synth_window_box, _box), 0)

            else:
                _height_1 = _synth_window_image.shape[0]
                _height_2 = _image.shape[0]
                _width_1 = _synth_window_image.shape[1]
                _width_2 = _image.shape[1]
                if _width_1 > _width_2:
                    # concat image
                    _diff_width = _width_1 - _width_2
                    _left_padding = np.random.randint(0, _diff_width + 1)
                    _right_padding = _diff_width - _left_padding
                    _top_padding = np.random.randint(30, 100)
                    _image = cv2.copyMakeBorder(_image, top=_top_padding, bottom=0, left=_left_padding,
                                                right=_right_padding, borderType=cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))
                    _synth_window_image = cv2.vconcat((_synth_window_image, _image))

                    # # concat label
                    # _label = cv2.copyMakeBorder(_label, top=_top_padding, bottom=0, left=_left_padding,
                    #                             right=_right_padding, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    # _synth_window_label = cv2.vconcat((_synth_window_label, _label))

                    # concat box [left,top,right,bottom]
                    _box = _box[:] + [_left_padding, _height_1 + _top_padding, _left_padding, _height_1 + _top_padding]
                    _synth_window_box = np.concatenate((_synth_window_box, _box), 0)
                else:
                    # concat image
                    _diff_width = _width_2 - _width_1
                    _left_padding = np.random.randint(0, _diff_width + 1)
                    _right_padding = _diff_width - _left_padding
                    _synth_window_image = cv2.copyMakeBorder(_synth_window_image, top=0, bottom=0,
                                                             left=_left_padding, right=_right_padding,
                                                             borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

                    _top_padding = np.random.randint(30, 100)
                    _image = cv2.copyMakeBorder(_image, top=_top_padding, bottom=0, left=0,
                                                right=0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    _synth_window_image = cv2.vconcat((_synth_window_image, _image))

                    # # concat label
                    # _synth_window_label = cv2.copyMakeBorder(_synth_window_label, top=0, bottom=0,
                    #                                          left=_left_padding, right=_right_padding,
                    #                                          borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    #
                    # _label = cv2.copyMakeBorder(_label, top=_top_padding, bottom=0, left=0,
                    #                             right=0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    # _synth_window_label = cv2.vconcat((_synth_window_label, _label))

                    # concat box
                    _synth_window_box = _synth_window_box[:] + [_left_padding, 0, _left_padding, 0]
                    _box = _box[:] + [0, _height_1 + _top_padding, 0, _height_1 + _top_padding]
                    _synth_window_box = np.concatenate((_synth_window_box, _box), 0)

        # 최종 합성 이미지에 패딩 추가
        sy_height, sy_width, _ = _synth_window_image.shape
        _hw = abs(sy_width - sy_height)
        if sy_height < sy_width:
            _left_padding = 100
            _right_padding = 100
            _top_padding = int(_hw / 2)
            _bottom_padding = int(_hw / 2)
        else:
            _left_padding = int(_hw / 2)
            _right_padding = int(_hw / 2)
            _top_padding = 100
            _bottom_padding = 100

        _synth_window_box = _synth_window_box[:] + [_left_padding, _top_padding, _left_padding, _top_padding]

        _synth_window_image = cv2.copyMakeBorder(_synth_window_image, top=_top_padding, bottom=_bottom_padding,
                                                 left=_left_padding, right=_right_padding,
                                                 borderType=cv2.BORDER_CONSTANT,
                                                 value=(255, 255, 255))
        # _synth_window_label = cv2.copyMakeBorder(_synth_window_label, top=_top_padding, bottom=_bottom_padding,
        #                                          left=_left_padding, right=_right_padding, borderType=cv2.BORDER_CONSTANT,
        #                                          value=(0, 0, 0))

        # resize bounding box
        h_ratio = self.image_size / _synth_window_image.shape[0]
        w_ratio = self.image_size / _synth_window_image.shape[1]

        # resize image
        _synth_window_image = cv2.resize(_synth_window_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # _synth_window_label = cv2.resize(_synth_window_label, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        _synth_window_label = cv2.bitwise_not(_synth_window_image)
        contour, hierachy = cv2.findContours(_synth_window_label[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _synth_window_label = cv2.drawContours(np.zeros_like(_synth_window_label), contour, -1, (255, 255, 255), 1)[:, :, 0:1]

        _synth_window_box = _synth_window_box[:] * [w_ratio, h_ratio, w_ratio, h_ratio]

        invert_image = cv2.bitwise_not(_synth_window_image)
        # print()
        # a = cv2.threshold(_synth_window_label,0.5,255,cv2.THRESH_BINARY)

        # 255 - white 0 - black
        # 원본 이미지에서 label이미지를 빼주고
        mean = int(np.mean(bg_image) / 2)

        number1 = np.ones_like(invert_image) * mean
        number2 = np.ones_like(invert_image) * 255
        invert_image = cv2.divide(invert_image, number2)
        invert_image = cv2.multiply(invert_image, number1)
        if np.random.choice([True, False], 1):
            bg_image = cv2.subtract(bg_image, invert_image)
        else:
            bg_image = cv2.add(bg_image, invert_image)

        # debug ######################################################################
        _debug_image = bg_image.copy()
        for _bbox in _synth_window_box:
            left = int(round(_bbox[0]))
            top = int(round(_bbox[1]))
            right = int(round(_bbox[2]))
            bottom = int(round(_bbox[3]))
            _debug_image = cv2.rectangle(_debug_image, (left, top), (right, bottom), (0, 0, 255), 1)
        ##############################################################################################

        return bg_image, _synth_window_box, _debug_image, _synth_window_label

    def _get_background(self,index):
        root = str(self.root)
        background_id = index % len(self.background_ids["1"])
        bg_ids = self.background_ids["1"][background_id]
        boxes, labels = self._get_annotation(self.background_ids["1"][background_id], 1, 'Background')
    
        window_image = cv2.imread(root + f"/Background/1/{bg_ids}", cv2.IMREAD_COLOR)
        box_id = np.random.randint(0,len(boxes))
        [left,top,right,bottom] = boxes[box_id]
        left = int(round(left))
        top = int(round(top))
        right = int(round(right))
        bottom = int(round(bottom))

        image = window_image[top:bottom,left:right]

        return image

    def _get_annotation(self,image_id,label,type):
        '''
            return box의 위치 , label
        '''
        if image_id.endswith('.bmp'):
            label_id = image_id.replace('.bmp','.lmk')
        elif image_id.endswith('.jpg'):
            label_id = image_id.replace('.jpg','.lmk')

        annotation_file = self.root / f"{type}/{label}/{label_id}"
        int_nBoxCnt, label_list, box_list = lmk_decoder(annotation_file)

        return (np.array(box_list,dtype=np.float32),
                np.array(label_list,dtype=np.float32))

    @staticmethod
    def _read_image_ids(path):
        ids = {}
        for _dir in os.listdir(path):
            if not _dir in ids:
                ids[_dir] = []
            cur_path = path / _dir
            for _file in os.listdir(cur_path):
                if _file.startswith("Image"):
                    if _file.endswith(".jpg") or _file.endswith(".bmp"):
                        ids[_dir].append(_file)

        return ids

    def _read_image(self, image_id,label):
        label_id = image_id.replace('.bmp', '.lmk')
        image_file = self.root / f"Font/{label}/{image_id}"
        label_file = self.root / f"Font/{label}/{label_id}"
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(str(label_file),cv2.IMREAD_GRAYSCALE)
        return image, label


class OCRDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/trainval.txt"
        self.ids = OCRDataset._read_image_ids(image_sets_file)

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default classes.")
            self.class_names = ('BACKGROUND',
                                '0', '1', '2', '3','4','5','6','7','8','9',
                                'A', 'B', 'C', 'D', 'E','F','G','H','I','J',
                                'K', 'L', 'M', 'N','O','P','Q','R','S','T',
                                'U', 'V', 'W','X','Y','Z')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels, f'Image_{image_id}.bmp'

    def __len__(self):
        return 30#len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self,image_id):
        '''
            return box의 위치 , label
        '''
        annotation_file = self.root / f"Images/Image_{image_id}.lmk"
        int_nBoxCnt, label_list, box_list = lmk_decoder(annotation_file)

        return (np.array(box_list,dtype=np.float32),
                np.array(label_list,dtype=np.float32))

    def _read_image(self, image_id):
        image_file = self.root / f"Images/Image_{image_id}.bmp"
        image = cv2.imread(str(image_file),cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image