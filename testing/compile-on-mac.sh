# sudo pip3 install pyinstaller
pyinstaller --onefile main.py
cp logic/config.yml dist/
mkdir dist/logic
cp logic/sysconfig.yml dist/logic/
cp sample/data-file.txt dist/
rm -r build
rm -r __pycache__
rm main.spec

