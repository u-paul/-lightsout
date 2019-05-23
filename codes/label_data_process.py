import os
import pandas as pd
import numpy as np
import csv



r1 = [1,2,3,4,7,8,11,12]
r2= list(range(15,37))
r3 = list(range(39,83))
r4 = [87,88]
r = r1+r2+r3+r4 


u = [6,9,13,37,84,87,89]

def gatherData():
	q = input("Want cvs? y or n?")
	sumTotal = []
	agree = []
	conflicts = []
	j= 1
	for i in range(0, len(r),2):
		n = r[i]
		df1= pd.read_csv('../dataSet/data4turk/labelled/clean/'+ str(n)+'labelled_data.csv')
		df2 = pd.read_csv('../dataSet/data4turk/labelled/clean/'+str(n+1)+'labelled_data.csv')
		df3= pd.merge(df1, df2, on='post')
		# print('Total number for file {} is {}: '.format(n, df3.shape))
		sumTotal.append(df3.shape[0])

		df_agree = df3.loc[df3['label_x'] == df3['label_y']]
		df_conflict = df3.loc[df3['label_x'] != df3['label_y']]
		df_agree = df_agree[['post', 'label_x']]
		df_conflict = df_conflict[['post']]
		agree.append(df_agree.shape[0])
		conflicts.append(df_conflict.shape[0])
		print('Conflicts for file {} and {} is {}: '.format(n, n+1, df_conflict.shape[0]))
		
		if q == 'y':
			df_agree.to_csv('../dataSet/data4turk/labelled/agree/'+str(j)+'agree.csv', index = False)
			df_conflict.to_csv('../dataSet/data4turk/labelled/conflicts/'+str(j)+'conflict.csv',header = False, index = False)

		j += 1
	print("Total obtained: ", sum(sumTotal))
	print(conflicts)
	print("Total agree: ", sum(agree))
	print("Total conflicts: ", sum(conflicts))



def unclaimCheck():
	tValue = []
	for i in range(len(u)):
		df= pd.read_csv('../dataSet/data4turk/labelled/clean/'+ str(u[i])+'labelled_data.csv')
		tValue.append(df.shape[0])


	print("The unvalidated data quantity: ", sum(tValue))



def makeBigFile():
	starting_folder = ('../dataSet/data4turk/labelled/agree/')
	files = []
	big_file = []

	for file in os.listdir(starting_folder):
		filename = os.fsdecode(file)
		files.append(filename)


	for i in range(len(files)):
		fname = starting_folder+files[i]
		df  = pd.read_csv(fname, index_col = None, header = 0)
		big_file.append(df) 

	big_file = pd.concat(big_file, axis = 0, ignore_index = True)
	print(big_file.shape)
	# big_file = pd.concat(big_file, axis = 0)
	# print(big_file.head())
	# print(big_file.shape[0])
	# print(big_file.shape)
	big_file.to_csv('../dataSet/data4turk/labelled/agree/overall.csv', columns= ['post', 'label_x'], index = False)


def labelChecker():

	df = pd.read_csv('../dataSet/data4turk/labelled/agree/overall.csv')
	# print(df.shape[0])
	nr = df.query('label_x == "general_not_relevant"').label_x.count()
	p1 = df.query('label_x == "communication-not-relevant"').label_x.count()
	print(p1)

# gatherData()
# unclaimCheck()
# makeBigFile()
labelChecker()





# df = df[['post']]
# print('Conflicts: ', df.shape)
# df.to_csv('../dataSet/data4turk/labelled/conflicts/'+str(j)+'conflict.csv', index = False, header = False)


# starting_folder = ('../dataSet/data4turk/labelled/conflicts/')

# files = []

# for file in os.listdir(starting_folder):
# 	filename = os.fsdecode(file)
# 	files.append(filename)
# c = []
# for i in range(len(files)):
# 	name = starting_folder + files[i]
# 	print('now reading: ', files[i])
# 	with open(name, 'r') as f:
# 		readCSV = csv.reader(f, delimiter=',')
# 		for row in readCSV:
# 			c.append(row)

# print(len(c))