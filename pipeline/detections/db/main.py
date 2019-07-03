from FDDBDataset import FDDBDataset
import cv2
import random


def test1_works_image():
    index_number_image = 40
    dataset = FDDBDataset()
    dataset.read('originalPics/', 'FDDB-folds/all-ellispeList.txt')
    images = dataset.data()['images']
    bboxes = dataset.data()['bboxes']
    box = bboxes[index_number_image][0]
    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    img = cv2.imread(images[index_number_image])
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    cv2.imshow('image', img)
    cv2.waitKey(0)

def get_data(train,val,test):
    dataset = FDDBDataset()
    dataset.read('originalPics/','all-ellispeList.txt')
    images = dataset.data()['images']
    bboxes = dataset.data()['bboxes']
    
    if (train+test+val>100):
        return "invalid input"
    
    random.shuffle(images)
    random.shuffle(bboxes)
    
    train=int(train/100*len(images))
    val=int(val/100*len(images))
    test=int(test/100*len(images))        
    
    train_images=images[0:train]
    train_bboxes=bboxes[0:train] 
    
    val_images=images[train:train+val]
    val_bboxes=bboxes[train:train+val]
    
    test_images=images[train+val:train+val+test]
    test_bboxes=bboxes[train+val:train+val+test]
    
    train_data=[train_images,train_bboxes]
    val_data=[val_images,val_bboxes]
    test_data=[test_images,test_bboxes]
    
    data=[train_data,val_data,test_data]

    return data


if __name__ == '__main__':
    test1_works_image()
