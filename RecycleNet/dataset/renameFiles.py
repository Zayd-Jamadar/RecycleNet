import os

cardboard = os.listdir('cardboard')
glass = os.listdir('glass')
metal = os.listdir('metal')
paper = os.listdir('paper')
plastic = os.listdir('plastic')

for count, filename in enumerate(cardboard):
	newName = 'cardboard'+str(count)+'.jpg'
	print(newName)
	os.rename('cardboard/'+filename, 'cardboard/'+newName)

for count, filename in enumerate(glass):
	newName = 'glass'+str(count)+'.jpg'
	print(newName)
	os.rename('glass/'+filename, 'glass/'+newName)

for count, filename in enumerate(metal):
	newName = 'metal'+str(count)+'.jpg'
	print(newName)
	os.rename('metal/'+filename, 'metal/'+newName)

for count, filename in enumerate(paper):
	newName = 'paper'+str(count)+'.jpg'
	print(newName)
	os.rename('paper/'+filename, 'paper/'+newName)

for count, filename in enumerate(plastic):
	newName = 'plastic'+str(count)+'.jpg'
	print(newName)
	os.rename('plastic/'+filename, 'plastic/'+newName)