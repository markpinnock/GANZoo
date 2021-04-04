import numpy as np
import os


FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/009_GAN_CT/Train/"
file_list = os.listdir(FILE_PATH)

for i in range(len(file_list)):
    if len(file_list[i]) == 33:
        # print(file_list[i])
        os.remove(FILE_PATH + file_list[i])
