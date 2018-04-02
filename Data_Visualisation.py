import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import dicom
import math

''' the following is simply the first pass of the dataset.
In this code I import the data, preprocess it so that the CT slices are homogenous across
all patients and then parse it through a 3DCNN.
This model can be replaced with other models too from the model folder'''

# firstly import the data (define the path where the scans are located)
data_directory= r'E:\DataSets\3DCNN_Cancer_detection\stage1'
patients= os.listdir(data_directory)

# the csv contains the same patient id and the corresponding labels (i.e. 1= cancer, 0= healthy)
patient_id=pd.read_csv(r'E:\DataSets\3DCNN_Cancer_detection\stage1_labels\stage1_labels.csv',index_col=0)

print(len(patients))

''' so I essentially have a list of patients and within that list of patients
I have a list of dicom files for each patient (i.e. slices). The no of slices vary
from patient to patient, so I want to take out a portion of these slices for all patients'''

pix_size= 100
Slice_count=40

#n-sized portions from list of slices (List)
def portions(List,n):
    count=0
    for i in range(0, len(List), n):
        if(count < Slice_count):
            yield List[i:i+n]
            count=count+1


def mean(List):
    return sum(List)/len(List)


def PreProcessData(patients,patient_id,Pixel_size,slice_count,visualise=False):
    label= patient_id.get_value(i,'cancer')
    path= os.path.join(data_directory,i)
    slices= [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    # resize the slice:
    slices= [cv2.resize(np.array(j.pixel_array),(pix_size,pix_size)) for j in slices]

    new_slices=[]

    portion_size = math.floor(len(slices) / Slice_count)

    for k in portions(slices,portion_size):
        k= list(map(mean, zip(*k)))
        new_slices.append(k) # these are the new slice portions

    print(len(new_slices))


    if visualise==True:
        figure= plt.figure()
        for num, j in enumerate(new_slices):
            images= figure.add_subplot(5,4,num+1)
            images.imshow(j)
        plt.show()

    if visualise==False:
        pass

# now this is were I will do the actual processing:

processed_data=[]

for i in enumerate(patients):
    try:
        ImageData,label= PreProcessData(i,patient_id,Pixel_size=pix_size, slice_count=Slice_count)
        processed_data.append([ImageData,label])
    # Reason why I want to try the above statement is, that for some patients, the labels are missing
    # So I will process the data above but for the ones I get an error, I just won't add them to my data list
    except KeyError as e:
        print(" This unlabelled data")
print('Processing complete and all {} patient data stored to memory'.format(len(patients)))
# save the file:
np.save('processed_data_{}_{}.npy'.format(pix_size,Slice_count),processed_data)
