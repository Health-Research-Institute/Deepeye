# DeepEye

Install VS Code: https://code.visualstudio.com/download
Install Python 3.9.13: https://www.python.org/downloads/release/python-3913/
Add C:\Users\User\AppData\Local\Programs\Python\Python39\ and C:\Users\User\AppData\Local\Programs\Python\Python39\Scripts\ to System Variable PATH
In VS Code perform the following steps:
	Choose Command Prompt for Terminal (next stages are according to https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)
	Create vitual environment - run: python -m venv deepeye 
	Activate vitual environment - run: source deepeye/bin/activate deepeye
	If you still have a (base) venv before prompt - run: conda deactivate
	Update pip - run: python -m pip install --upgrade pip
 	Install virtual environment
	Install Numpy package - run: pip install numpy
	Install Pandas package - run: pip install pandas
	Install Matplotlib - run: pip install matplotlib 
	Install OpenCV - run: pip install opencv-python
	If you have GPU: install CUDA from here
	https://developer.nvidia.com/cuda-downloads
	Install Pytorch (see https://pytorch.org/get-started/locally/#windows-pip) 
	without GPU - run: pip3 install torch torchvision torchaudio
	with GPU - run: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 (or cu118 based on current version)

	General knowledge: to get the Python IDLE in VS Code environment - run "python" at cmd window.
	
	For Keras installation:
		Install Scipy package - run: pip install scipy
		Install Scikit-learn package - run: pip install -U scikit-learn
		Install Seaborn package - run: pip install seaborn
		Install Tensorflow package - run: pip install tensorflow
		Install Keras package - run: pip install keras
		
	For manual labeling of points
		pip install mpl_point_clicker
