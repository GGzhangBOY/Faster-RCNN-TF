from PIL import Image
import os
import os.path
import numpy as np 
import copy

class breakPoint:
    line_index = 0
    file_index = 0
    def __init__(self,l_index = 0,f_index = 0):
        self.line_index = l_index
        self.file_index = f_index

def file_recursion(path_input,file_list):
    file_list = file_list
    if(os.path.exists(path_input)):
        path_dir = os.path.abspath(path_input)
        for i in os.listdir(path_dir):
            path_i = os.path.join(path_dir,i)
            if(os.path.isfile(path_i)):
                if(path_i.find("ellipseList")):
                    file_list.append(path_i)
            else:
                file_recursion(path_i,file_list)
    else:
        print("ERROR: The path"+path_input+ "dose not exists!")

    return file_list

def open_pic(path_input):
    img = Image.open(path_input)
    pic = np.array(img)

    if(len(pic.shape)>2):
        channels = pic.shape[2]
    else:
        channels = 1

    dimx = pic.shape[0]
    dimy = pic.shape[1]

    return pic,channels,dimx,dimy

class Data_label:
    img_name = ""
    Data = 0
    gt_pos = []
    dimx = 0
    dimy = 0
    num_faces = 1
    pic_channels = 0

    def __process_to_rect(self,get_pos):
        result = []
        for buff in get_pos:
            x = buff[2]
            y = buff[3]
            h = 2*buff[0]
            w = 2*buff[1]
            result.append([x,y,w,h])
        
        return result

    def __init__(self,img_Name,Data,dimx,dimy,gt_pos,num_faces,pic_channels):
        self.img_name = img_Name
        self.Data=Data
        self.gt_pos.append(self.__process_to_rect(gt_pos))
        self.num_faces = num_faces
        self.dimx = dimx
        self.dimy = dimy
        self.pic_channels = pic_channels

class Dataset:
    
    total_batches = 0
    batch_size = 1
    dataset_year = "2002"
    num_batch_processed = 0
    __Lines_storer = []
    __Files_storer = []
    __is_break_point_activate = False
    __is_reading_file_changed = True
    break_point = breakPoint()

    def __init__(self, batch_size_in=1, dataset_year="2002"):
        self.batch_size = batch_size_in
        self.dataset_year = dataset_year
        
    def __read_data(self):
        Data = []
        Datas = []
        num_pic_processed = 0
        lines = []
        flag_text = False 
        num_gt = 0
        file_list = file_recursion('C:\\Dev_env\\Tensorflow_Python\\FDDB-folds\\',[])

        i = 0
        i_l = 0

        while(i < len(file_list)):
            file = file_list[i]
            if(self.__is_reading_file_changed == True):
                f = open(file,'r')
                for line in f:
                    lines.append(line)
                self.__is_reading_file_changed = False
            
            else:
                lines = self.__Lines_storer

            while(i_l<len(lines)):
                line = lines[i_l]
                if(line.find('img')>0):
                    line = line.replace('\n','')
                    p_name = line+'.jpg'
                    flag_text = True
                    lines.remove(lines[0])
                    i_l = 0
                    continue

                if(flag_text == True):
                    num_gt = int(line)
                    flag_text = False
                    lines.remove(lines[0])
                    i_l = 0
                    continue

                if(num_gt != 0):
                    buff = []
                    Data_buff = line.split(' ')
                    
                    for i in range(4):
                        buff.append(float(Data_buff[i]))
                    Data.append(buff)
                    num_gt-=1
                    lines.remove(lines[0])
                    i_l = 0
                
                if((num_pic_processed>=0)&(num_gt == 0)):
                    img_Path = 'C:\\Dev_env\\Tensorflow_Python\\' + p_name
                    img_Path = img_Path.replace('/','\\')
                    pic,channels,dimx,dimy = open_pic(img_Path)
                    Datas.append(Data_label(p_name,pic,dimx,dimy,Data,num_gt,channels))
                    num_pic_processed += 1
                    Data = []
                    if(num_pic_processed<self.batch_size):
                        continue
                    else:
                        self.num_batch_processed += 1
                        self.__Lines_storer = lines
                        return Datas

            self.__is_reading_file_changed = True
            file_list.remove(file_list[0])
            i = 0
            
    def next_batch(self):
        data_index = []
        data_labels = []
        if(self.__is_break_point_activate == False):
            Datas = self.__read_data()
            self.__is_break_point_activate  = True
        else:
            Datas = self.__read_data()

        pic = Datas.pop()
        data_index = pic.Data
        data_labels = copy.deepcopy(pic.gt_pos)
        pic.gt_pos.remove(pic.gt_pos[0])

        return data_index,data_labels


if(__name__ == '__main__'):
    test = Dataset()
    _,label1 = test.next_batch()

    print(label1)

    _,label2 = test.next_batch()

    print(label2)

    _,label3 = test.next_batch()

    print(label3)