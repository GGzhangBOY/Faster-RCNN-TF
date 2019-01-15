from PIL import Image
import os
import os.path
import numpy as np 
import copy

def file_recursion(path_input,file_list):
    file_list = file_list
    if(os.path.exists(path_input)):
        path_dir = os.path.abspath(path_input)
        for i in os.listdir(path_dir):
            path_i = os.path.join(path_dir,i)
            if(os.path.isfile(path_i)):
                if(path_i.find("ellipseList")!=-1):
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

    dimx = pic.shape[1]
    dimy = pic.shape[0]

    pic = pic.tolist()
    return pic,channels,dimx,dimy


class Data_label:
    img_name = ""
    gt_pos = []
    num_faces = 1
    label = 1

    def __process_to_rect(self,get_pos):
        result = []
        for buff in get_pos:
            x1 = buff[2] - buff[0]
            y1 = buff[3] - buff[1]
            x2 = buff[2] + buff[0]
            y2 = buff[3] + buff[1]
            result.append([x1,y1,x2,y2])
        
        return result
    
    def __init__(self,img_Name,gt_pos,num_faces,label):
        self.img_name = img_Name
        self.gt_pos.append(self.__process_to_rect(gt_pos))
        self.num_faces = num_faces
        self.label = label

class Dataset_cus:

    def __init__(self, batch_size_in=1, dataset_year="2002"):
        self.batch_size = batch_size_in
        self.dataset_year = dataset_year
        self.Datas = self.__read_data()
        self.ftrain = self.Datas[:int(0.9*len(self.Datas))]
        self.ftest = self.Datas[int(0.9*len(self.Datas)):]
        self.num_train_process = 0
        self.num_test_process = 0

    def __read_data(self):
        Data = []
        Datas = []
        num_pic_processed = 0
        lines = []
        flag_text = False 
        num_gt = 0
        current_label = 0
        file_list = file_recursion("Faster-RCNN-TF\FDDB-folds",[])

        i = 0
        i_l = 0
        num_txt = 0
        while(num_txt < len(file_list)):
            file_name = file_list[num_txt]
            f = open(file_name,'r')
            for line in f:
                lines.append(line)

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
                    
                    for i in range(6):
                        if(i == 2 or i == 5):
                            continue
                        elif(i == 6):
                            current_label = float(Data_buff[i])
                        else:
                            buff.append(float(Data_buff[i]))

                    Data.append(buff)
                    num_gt-=1
                    lines.remove(lines[0])
                    i_l = 0
                
                if((num_pic_processed>=0)&(num_gt == 0)):
                    img_Path = 'Faster-RCNN-TF\\' + p_name
                    img_Path = img_Path.replace('/','\\')
                    Datas.append(Data_label(p_name,Data,num_gt,current_label))
                    num_pic_processed += 1

            num_txt+=1
        return Datas
            
    def next_batch(self,batch_size = 1,mod = "train"):
        datas = []
        labels = []
        gt_boxs = []
        sources = []
        labels_rt = []
        gt_boxs_rt = []

        if(mod == "train"):
            if((self.num_train_process+batch_size)<len(self.ftrain)):
                for i in range(batch_size):
                    datas.append(self.ftrain[self.num_train_process+i].img_name)
                    labels.append(self.ftrain[self.num_train_process+i].label)
                    gt_boxs.append(self.ftrain[self.num_train_process+i].gt_pos)
                self.num_train_process+=batch_size
            else:
                for i in range(len(self.ftrain)-self.num_train_process):
                    datas.append(self.ftrain[self.num_train_process+i].img_name)
                    labels.append(self.ftrain[self.num_train_process+i].label)
                    gt_boxs.append(self.ftrain[self.num_train_process+i].gt_pos)
                self.num_train_process =0
        if(mod == "test"):
            if((self.num_test_process+batch_size)<len(self.ftest)):
                for i in range(batch_size):
                    datas.append(self.ftest[self.num_test_process+i].img_name)
                    labels.append(self.ftest[self.num_test_process+i].label)
                    gt_boxs.append(self.ftest[self.num_test_process+i].gt_pos)
                self.num_test_process+=batch_size
            else:
                for i in range(len(self.ftest)-self.num_test_process):
                    datas.append(self.ftest[self.num_test_process+i].img_name)
                    labels.append(self.ftest[self.num_test_process+i].label)
                    gt_boxs.append(self.ftest[self.num_test_process+i].gt_pos)

        for fname in datas:
            img = Image.open(fname)
            sources.append(np.array(img))
        for label in labels:
            labels_rt.append([int(label)])         
        for gt_box in gt_boxs:
            gt_boxs_rt.append(gt_box)
            
        dimx = np.array(sources[0]).shape[1]
        dimy = np.array(sources[0]).shape[2]
        channels = np.array(sources[0]).shape[3]

        return sources,labels_rt,gt_boxs_rt,dimx,dimy,channels


if(__name__ == '__main__'):
    test = Dataset_cus()
    _,label1,_,_,_ = test.next_batch()

    print(label1,len(np.array(label1).shape))

    _,label2,_,_,_ = test.next_batch()

    print(label2,np.array(label2).shape)

    _,label3,_,_,_ = test.next_batch()

    print(label3,np.array(label3).shape)