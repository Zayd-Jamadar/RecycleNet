import os
import random

cardboard = os.listdir('dataset/cardboard')
glass = os.listdir('dataset/glass')
metal = os.listdir('dataset/metal')
paper = os.listdir('dataset/paper')
plastic = os.listdir('dataset/plastic')

print('Cardboard: {length:4d} {seven} {thirty}'.format(length=len(cardboard),
	seven=int(len(cardboard)*0.7),
	thirty=(((len(cardboard))-int(len(cardboard)*0.7)))))

print('Glass: {length:8d} {seven} {thirty}'.format(length=len(glass),
	seven=int(len(glass)*0.7),
	thirty=(((len(glass))-int(len(glass)*0.7)))))

print('Metal: {length:8d} {seven} {thirty}'.format(length=len(metal),
	seven=int(len(metal)*0.7),
	thirty=(((len(metal))-int(len(metal)*0.7)))))

print('Paper: {length:8d} {seven} {thirty}'.format(length=len(paper),
	seven=int(len(paper)*0.7),
	thirty=(((len(paper))-int(len(paper)*0.7)))))

print('Plastic: {length:6d} {seven} {thirty}'.format(length=len(plastic),
	seven=int(len(plastic)*0.7),
	thirty=(((len(plastic))-int(len(plastic)*0.7)))))

CUR_DIRECTORY = os.path.abspath(os.getcwd())

for _ in range(int(len(cardboard)*0.7)):
	choice = random.choice(cardboard)
	os.rename(CUR_DIRECTORY+'/dataset/cardboard/'+choice,
		CUR_DIRECTORY+'/dataset/train/cardboard/'+choice)
	cardboard = os.listdir('dataset/cardboard')

# for _ in range(int(len(glass)*0.7)):
# 	choice = random.choice(glass)
# 	os.rename('/dataset/glass/'+choice+)

# for _ in range(int(len(metal)*0.7)):
# 	choice = random.choice(metal)
# 	os.rename('/dataset/metal/'+choice+)

# for _ in range(int(len(paper)*0.7)):
# 	choice = random.choice(paper)
# 	os.rename('/dataset/paper/'+choice+)

# for _ in range(int(len(plastic)*0.7)):
# 	choice = random.choice(plastic)
# 	os.rename('/dataset/plastic/'+choice+)