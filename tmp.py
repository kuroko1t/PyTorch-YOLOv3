import xml.etree.ElementTree as ET
from pprint import pprint
import glob
import os

class XML:
    def __init__(self, path):
        self.path = path
        self.extract_xml()
        self.to_yolo()

    def to_yolo(self):
        """
        scale x, y to [0,1]
        """
        self.xcenter = (self.xmin + self.xmax) / 2.
        self.ycenter = (self.ymin + self.ymax) / 2.
        self.xcenter = self.xcenter / self.width
        self.ycenter = self.ycenter / self.width
        
    def extract_xml(self):
        tree = ET.parse(self.path)
        root = tree.getroot()
        for child in root:
            if child.tag == 'filename':
                self.imagename = child.text.replace('JPG', 'jpg')
                print(self.imagename)
            if child.tag == 'size':
                for cchild in child:
                    if cchild.tag == "width":
                        self.width = int(cchild.text)
                    elif cchild.tag == "height":
                        self.height = int(cchild.text)
                    elif cchild.tag == "depth":
                        self.depth = int(cchild.text)
            elif child.tag == 'object':
                for cchild in child:
                    if cchild.tag == "name":
                        self.class_name  = cchild.text
                    elif cchild.tag == 'bndbox':
                        for ccchild in cchild:
                            if ccchild.tag == 'xmin':
                                self.xmin = float(ccchild.text)
                            elif ccchild.tag == 'ymin':
                                self.ymin = float(ccchild.text)
                            elif ccchild.tag == 'xmax':
                                self.xmax = float(ccchild.text)
                            elif ccchild.tag == 'ymax':
                                self.ymax = float(ccchild.text)

class Annotation:
    def __init__(self, annotation_path, image_path):
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.annotate()
 
    def annotate(self):
       xml_list = glob.glob(f'{self.annotation_path}/*.xml')
       self.annotations = []
       for xml_file in xml_list:
           at = XML(xml_file)
           self.annotations.append(at)

    def create_annotate_file(self, path):
        #class_name = {'kasu':0, '002_kasu':1, '001_dakon':2, 'dakon':3}
        class_name = {'kasu':0, '002_kasu':0, '001_dakon':1, 'dakon':1}
        train_url = f'./data/custom/train.txt'
        val_url = f'./data/custom/valid.txt'
        #if os.path.exists(train_url):
        #    os.remove(train_url)
        #    os.remove(val_url)
        
        for at in self.annotations:
            idx = class_name[at.class_name]
            file_name = at.imagename.replace('.jpg', '.txt')
            with open(f'{path}/{file_name}', 'w') as f:
                f.write(f'{idx} {at.xcenter} {at.ycenter} {at.width} {at.height}')
            with open(train_url, 'a') as f:
                f.write(f'{self.image_path}/{at.imagename}\n')
                
            
if __name__ == "__main__":
    at = Annotation('/home/kurosawa/share/Annotations/',
                    '/home/kurosawa/share/Images/')
    #at.create_annotate_file('./data/custom/labels/')
