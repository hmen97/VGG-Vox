import wget
import os

l = ['http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacaa',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacac',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacad',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacae',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacaf',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacag',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacah',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aacab',
	'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip',
	]

user = "voxceleb1911"
password = "7ddc50p9"


for url in l:
	os.system("wget --user voxceleb1911 --password 7ddc50p9 " + url)

