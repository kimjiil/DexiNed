import struct
import cv2
import glob

def my_read(file,count):
    _temp = file.read(count)

    _byte = ''
    for i in _temp:
        # print(hex(i),end=' ')
        if len(hex(i)) == 3:
            temp_str = hex(i).replace('0x', r'\x0')  #0xFF
        else:
            temp_str = hex(i).replace('0x', r'\x')
        _byte += temp_str
    return _byte

def lmk_decoder(path):

    file = open(path,'rb')
    _encoding = file.read(2)
    if _encoding == b'\xff\xfe': #utf-16 인코딩 확인
        _byte = file.read(2)
        _,_string_size = struct.unpack('<BB',_byte)
    else:
        print('file error : encoding format is not utf-16le')
        return -1;

    pemtron_str = ''

    for i in range(_string_size):
        _str_byte = file.read(2)  #  example : \x43\x34
        _decode_str = _str_byte.decode('utf-16')

        pemtron_str += _decode_str

    _encoding = file.read(2)
    if _encoding == b'\xff\xfe':  # utf-16 인코딩 확인
        _byte = file.read(2)
        _, _string_size = struct.unpack('<BB', _byte)
    else:
        print('file error : encoding format is not utf-16le')
        return -1;

    version_str = ''

    for i in range(_string_size):
        _str_byte = file.read(2)
        _decode_str = _str_byte.decode('utf-16')

        version_str += _decode_str

    byte_headerformat = file.read(4) #int형
    headerformat_value, = struct.unpack('<I',byte_headerformat)

    _byte_header = file.read(headerformat_value)
    int_length , int_nBoxCnt = struct.unpack('<II',_byte_header)


    _encoding = file.read(2)
    if _encoding == b'\xff\xfe':  # utf-16 인코딩 확인
        _byte = file.read(2)
        _, _string_size = struct.unpack('<BB', _byte)
    else:
        print('file error : encoding format is not utf-16le')
        return -1;

    tmp_header_str = ''

    for i in range(_string_size):
        _str_byte = file.read(2)
        _decode_str = _str_byte.decode('utf-16')

        tmp_header_str += _decode_str

    ############# Rect Infomatrion decoding #############
    
    # if int_nBoxCnt > 1:
    #     print(Image_path,int_nBoxCnt)
    label_list = []
    box_list = []
    for i in range(int_nBoxCnt):
        
        _byte_int_dataformat = file.read(4)
        if _byte_int_dataformat == b'':
            break
        int_dataformat, = struct.unpack('<I',_byte_int_dataformat)

        _byte_Rect_info = file.read(int_dataformat)
        _left,_top,_right,_bottom,_label = struct.unpack('<IIIII',_byte_Rect_info)

        if _label >36:
            # Image_path = path._str.replace('lmk','bmp')
            # window_image = cv2.imread(Image_path, cv2.IMREAD_GRAYSCALE)
            # cv2.rectangle(window_image, (_left, _top), (_right, _bottom), (255, 0, 0))
            # cv2.imshow('test',window_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            break
        label_list.append(_label)


        box_list.append([_left,_top,_right,_bottom])
        # print('----------- box info {} ------------'.format(i))
        # print('Left : ',_left
        #       ,'Right : ',_right)
        # print('Top : ',_top,'Bottom : ',_bottom)
        # print('Label - ',_label)
        

    file.close()

    return int_nBoxCnt, label_list ,box_list

def lmk_decoder_viewer(path):

    path = path.replace('\\','/')
    file = open(path,'rb')
    _encoding = file.read(2)
    if _encoding == b'\xff\xfe': #utf-16 인코딩 확인
        _byte = file.read(2)
        _,_string_size = struct.unpack('<BB',_byte)
    else:
        print('file error : encoding format is not utf-16le')
        return -1;

    pemtron_str = ''

    for i in range(_string_size):
        _str_byte = file.read(2)  #  example : \x43\x34
        _decode_str = _str_byte.decode('utf-16')

        pemtron_str += _decode_str

    _encoding = file.read(2)
    if _encoding == b'\xff\xfe':  # utf-16 인코딩 확인
        _byte = file.read(2)
        _, _string_size = struct.unpack('<BB', _byte)
    else:
        print('file error : encoding format is not utf-16le')
        return -1;

    version_str = ''

    for i in range(_string_size):
        _str_byte = file.read(2)
        _decode_str = _str_byte.decode('utf-16')

        version_str += _decode_str

    byte_headerformat = file.read(4) #int형
    headerformat_value, = struct.unpack('<I',byte_headerformat)

    _byte_header = file.read(headerformat_value)
    int_length , int_nBoxCnt = struct.unpack('<II',_byte_header)


    _encoding = file.read(2)
    if _encoding == b'\xff\xfe':  # utf-16 인코딩 확인
        _byte = file.read(2)
        _, _string_size = struct.unpack('<BB', _byte)
    else:
        print('file error : encoding format is not utf-16le')
        return -1;

    tmp_header_str = ''

    for i in range(_string_size):
        _str_byte = file.read(2)
        _decode_str = _str_byte.decode('utf-16')

        tmp_header_str += _decode_str

    ############# Rect Infomatrion decoding #############
    Image_path = path.replace('lmk','bmp')
    # if int_nBoxCnt > 1:
    #     print(Image_path,int_nBoxCnt)
    label_list = []
    for i in range(int_nBoxCnt):
        window_image = cv2.imread(Image_path, cv2.IMREAD_GRAYSCALE)
        height = window_image.shape[0]
        _byte_int_dataformat = file.read(4)
        int_dataformat, = struct.unpack('<I',_byte_int_dataformat)

        _byte_Rect_info = file.read(int_dataformat)
        _left,_top,_right,_bottom,_label = struct.unpack('<IIIII',_byte_Rect_info)

        label_list.append(_label)
        print('----------- box info {} ------------'.format(i))
        print('Left : ',_left
              ,'Right : ',_right)
        print('Top : ',_top,'Bottom : ',_bottom)
        print('Label - ',_label)
        cv2.rectangle(window_image, (_left, _top), (_right, _bottom), (0, 0, 255))
        cv2.imshow('test',window_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    file.close()

if __name__ == '__main__':
    lmk_decoder_viewer(r'D:\gajeon_210830\1/Image_5950.lmk')
    # lmk_decoder_viewer('D:/_work_space/Data_Collection/김해 행성/김해 행성 데이터/라벨링 완료 데이터/20210427_fix (2)-작업완료/R10K_1608/20210427125048/Image_900.lmk')