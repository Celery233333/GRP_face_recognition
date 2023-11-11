from face import*
from utils import*
import os
import pytest


celery = face()


# Test same people     
def test_face1():
     testface1 = face()
     face_list = os.listdir("test/test1/testa")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          testface1.add_dataset("test/test1/testa/"+myface)
          if testface1.compare("test/test2/testa.jpg") == 1:
               count = 1
     assert count ==1

# Test same people  
def test_face2():
     testface2 = face()
     face_list = os.listdir("test/test1/testb")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          testface2.add_dataset("test/test1/testb/"+myface)
          if testface2.compare("test/test2/testb.jpg") == 1:
               count = 1
     assert count ==1


# Test similar but different people     
def test_face3():
     testface3 = face()
     face_list = os.listdir("test/test1/testc")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          testface3.add_dataset("test/test1/testc/"+myface)
          if testface3.compare("test/test2/testc.jpg") == 1:
               count = 1
     assert count ==0


# Test similar but different people     
def test_face4():
     testface4 = face()
     face_list = os.listdir("test/test1/testd")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          testface4.add_dataset("test/test1/testd/"+myface)
          if testface4.compare("test/test2/testd.jpg") == 1:
               count = 1
     assert count ==0
     

# Test glass
def test_glass1():
     testglass1 = face()
     face_list = os.listdir("test/testglass")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          if testglass1.glass_detect("test/testglass/"+myface) == 0:
               count = count + 1
     assert count <= len(face_list)/3
     

# Test no glass
def test_glass2():
     testglass2 = face()
     face_list = os.listdir("test/testnoglass")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          if testglass2.glass_detect("test/testnoglass/"+myface) == 1:
               count = count + 1
     assert count == 0
     
     
# Test smile
def test_smile1():
     testsmile1 = face()
     face_list = os.listdir("test/smile")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          if testsmile1.smile_detect("test/smile/"+myface) == 0:
               count = count + 1
     assert count <= 1
     

# Test no smile
def test_smile2():
     testsmile2 = face()
     face_list = os.listdir("test/nosmile")
     count = 0

        # check the dataset
     if len(face_list) == 0:
          exit("dataset is empty")

        # encoding all the images from image folder
     for myface in face_list:
          # testface2.add_dataset("test/test1/testd/"+myface)
          if testsmile2.smile_detect("test/nosmile/"+myface) == 1:
               count = count + 1
     assert count <= 1