import os
import pandas as pd

file2work = [ 'harvey']
# 'florence','irma', 'maria', 'matthew', 'michael', 'sandy',
for ii in file2work:

	starting_folder = ('../dataSet/data4turk/hurricane/'+ ii + '/')
	save_folder = ('../dataSet/data4turk/hurricane/'+ ii + '/')

	files = []
	for file in os.listdir(starting_folder):
		filename = os.fsdecode(file)
		files.append(filename)

	save_files = [w.replace('xls', 'csv') for w in files]

	for i in range(len(files)):
		try:
			name = starting_folder+files[i]
			df = pd.read_excel(starting_folder+files[i])
			df.to_csv(save_folder+save_files[i],index = False)
		except Exception as e:
			print(name)
			print(e)

