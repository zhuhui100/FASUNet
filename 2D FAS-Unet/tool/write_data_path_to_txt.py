import os


dir_name = 'test_304'

txt = '../dataset_2D/txt/' + dir_name + '.txt'
image_dir = './dataset_2D/' + dir_name + '/image/'
gt_dir = './dataset_2D/' + dir_name + '/gt/'


nameList = []
for filename in os.listdir("." + image_dir):
    nameList.append(filename)
nameList = sorted(nameList, key=lambda x: x)

with open(txt, 'w') as f:
    count = 0
    
    for filename in nameList:



        f.write( image_dir + filename)
        f.write(",")


        f.write( gt_dir + filename.replace("Patient", "GT"))
        f.write('\n')
        count += 1
    
    print('image number: ', count)


