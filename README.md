# DeepEye

Install VS 2019
Install VS Code: https://code.visualstudio.com/download
Install Python 3.9.13: https://www.python.org/downloads/release/python-3913/
Add C:\Users\User\AppData\Local\Programs\Python\Python39\ and C:\Users\User\AppData\Local\Programs\Python\Python39\Scripts\ to System Variable PATH
In VS Code perform the following steps:
	Choose Command Prompt for Terminal (next stages are according to https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)
	Update pip - run: python -m pip install --upgrade pip
	Install Numpy package - run: pip install numpy
	Install Matplotlib - run: pip install matplotlib 
	Install OpenCV - run: pip install opencv-python
	If you have GPU: install CUDA from here
	https://developer.nvidia.com/cuda-downloads
	Install Pytorch (see https://pytorch.org/get-started/locally/#windows-pip) 
	without GPU - run: pip3 install torch torchvision torchaudio
	without GPU - run: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 (or cu118 based on current version)