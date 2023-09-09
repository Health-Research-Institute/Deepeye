# DeepEye

Install VS Code: https://code.visualstudio.com/download
Install Python 3.11.4 (or the last version): https://www.python.org/downloads/ (if necessary)
Add C:\Users\User\AppData\Local\Programs\Python\Python39\ and C:\Users\User\AppData\Local\Programs\Python\Python39\Scripts\ to System Variable PATH

Install Conda (according to machine):
	For Windows: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
	For Mac: https://docs.conda.io/projects/conda/en/4.6.0/user-guide/install/macos.html
	Make sure you have the latest version of Conda: conda update -n base -c defaults conda

If you have GPU (Windows or Linux), install CUDA: https://developer.nvidia.com/cuda-downloads

At command prompt (for example, in VS Code environment):
	Create vitual environment - run: conda create -n deepeye python=3.11 pip 
	Activate vitual environment - run: conda activate deepeye
	Update pip - run: python -m pip install --upgrade pip

	For manual labeling of retina layers at https://www.makesense.ai/:
		Install mpl_point_clicker: pip install mpl_point_clicker --upgrade

 	For json_mask_reading.ipynb:
		Install Numpy package - run: pip install numpy --upgrade
		Install PIL package - run: pip install pillow --upgrade
		Install OpenCV - run: pip install opencv-python --upgrade
		Install Scikit-learn package - run: pip install scikit-learn --upgrade
		Install Matplotlib - run: pip install matplotlib --upgrade

	For multiclass_segmentation.ipynb:
		Install Tensorflow package - run: pip install tensorflow --upgrade (Keras and tensorflow-macos installations are included)
		For Keras on Mac:
			pip install tensorflow-metal --upgrade
			conda install -c apple tensorflow-deps=2.10.0 (may not install due to incompatibility, but may not be necessary)
	
	For DenseNet_Model.ipynb:
		Install Pytorch:
			For Windows (see https://pytorch.org/get-started/locally/#windows-pip) 
				Without GPU - run: pip3 install torch torchvision torchaudio
				With GPU - run: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 (or cu118 based on current version)
			For Mac with M1/M2:
				Run: pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
					 or: conda install pytorch torchvision torchaudio -c pytorch-nightly
		Install dotenv: pip install python-dotenv --upgrade

General knowledge: to get the Python IDLE in VS Code environment - run "python" at cmd window.
