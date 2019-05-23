import csv
import json
import glob
import os

# csvfile = open('../dataSet/data4turk/hurricane/2label/tweet-1000-set.csv', 'r')
# jsonfile = open('../dataSet/data4turk/hurricane/2label/1000.json', 'w')


# fieldnames = ("externalId","data",)
# reader = csv.DictReader( csvfile, fieldnames)
# out = json.dumps( [ row for row in reader ] )
# jsonfile.write(out)




starting_folder = ('../dataSet/data4turk/hurricane/2label/csvs/')
save_folder = ('../dataSet/data4turk/hurricane/2label/jsons/')

files = []
for file in os.listdir(starting_folder):
	filename = os.fsdecode(file)
	files.append(filename)

save_files = [w.replace('csv', 'json') for w in files]
for i in range(len(files)):
	try:
		csvfile = open(starting_folder+files[i], 'r')
		jsonfile = open(save_folder+save_files[i], 'w')
		fieldnames = ("externalId","data",)
		reader = csv.DictReader( csvfile, fieldnames)
		out = json.dumps( [ row for row in reader ] )
		jsonfile.write(out)
	except Exception as e:
		print(csvfile)
		print(e)
