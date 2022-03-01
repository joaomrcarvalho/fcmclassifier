# make sure the zip package is installed
# it isn't, by default, on WSL
# this will ask you for your sudo password
# if you're not an admin on the computer you're using, delete this line
sudo apt install zip

# download the sample data
wget https://joaomrcarvalho.github.io/datasets/fcmclassifier_sample_data.zip

# unzip it to sample_data folder
unzip fcmclassifier_sample_data.zip

# output quantization files for "quantized_data" folder
# and attempt to perform classification
python fcmclassifier.py sample_data quantized_data -a 17 -k 25 -d 3