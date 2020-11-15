import os
import random

cardboard = os.listdir('dataset/cardboard')
glass = os.listdir('dataset/glass')
metal = os.listdir('dataset/metal')
paper = os.listdir('dataset/paper')
plastic = os.listdir('dataset/plastic')

seven_cardboard = int(len(cardboard))
print('Cardboard: {length:4d} {seven} {thirty}'.format(length=seven_cardboard,
	seven=int(len(cardboard)*0.7),
	thirty=(((len(cardboard))-int(len(cardboard)*0.7)))))

seven_glass = int(len(glass))
print('Glass: {length:8d} {seven} {thirty}'.format(length=seven_glass,
	seven=int(len(glass)*0.7),
	thirty=(((len(glass))-int(len(glass)*0.7)))))

seven_metal = int(len(metal))
print('Metal: {length:8d} {seven} {thirty}'.format(length=seven_metal,
	seven=int(len(metal)*0.7),
	thirty=(((len(metal))-int(len(metal)*0.7)))))

seven_paper = int(len(paper))
print('Paper: {length:8d} {seven} {thirty}'.format(length=seven_paper,
	seven=int(len(paper)*0.7),
	thirty=(((len(paper))-int(len(paper)*0.7)))))

seven_plastic = int(len(plastic))
print('Plastic: {length:6d} {seven} {thirty}'.format(length=seven_plastic,
	seven=int(len(plastic)*0.7),
	thirty=(((len(plastic))-int(len(plastic)*0.7)))))


CUR_DIRECTORY = os.path.abspath(os.getcwd())

if(len(os.listdir('dataset/train/cardboard')) != 192):
	for _ in range(int(seven_cardboard*0.7)):
		choice = random.choice(cardboard)
		cardboard.remove(choice)
		os.rename(CUR_DIRECTORY+'/dataset/cardboard/'+choice,
			CUR_DIRECTORY+'/dataset/train/cardboard/'+choice)
	print(len(os.listdir('dataset/train/cardboard')))

if(len(os.listdir('dataset/train/glass')) != 178):
	for _ in range(int(seven_glass*0.7)):
		choice = random.choice(glass)
		glass.remove(choice)
		os.rename(CUR_DIRECTORY+'/dataset/glass/'+choice,
		CUR_DIRECTORY+'/dataset/train/glass/'+choice)
	print(len(os.listdir('dataset/train/glass')))

if(len(os.listdir('dataset/train/metal')) != 144):
	for _ in range(int(seven_metal*0.7)):
		choice = random.choice(metal)
		metal.remove(choice)
		os.rename(CUR_DIRECTORY+'/dataset/metal/'+choice,
		CUR_DIRECTORY+'/dataset/train/metal/'+choice)
	print(len(os.listdir('dataset/train/metal')))

if(len(os.listdir('dataset/train/paper')) != 196):
	for _ in range(int(seven_paper*0.7)):
		choice = random.choice(paper)
		paper.remove(choice)
		os.rename(CUR_DIRECTORY+'/dataset/paper/'+choice,
		CUR_DIRECTORY+'/dataset/train/paper/'+choice)
	print(len(os.listdir('dataset/train/paper')))

if(len(os.listdir('dataset/train/plastic')) != 340):
	for _ in range(int(seven_plastic*0.7)):
		choice = random.choice(plastic)
		plastic.remove(choice)
		os.rename(CUR_DIRECTORY+'/dataset/plastic/'+choice,
		CUR_DIRECTORY+'/dataset/train/plastic/'+choice)
	print(len(os.listdir('dataset/train/plastic')))

