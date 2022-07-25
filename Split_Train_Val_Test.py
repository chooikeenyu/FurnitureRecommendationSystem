import splitfolders

input_folder = 'images2'

splitfolders.ratio(input_folder,output="furniture-2022",seed=42,ratio=(0.7,0.2,0.1),
                   group_prefix=None)

