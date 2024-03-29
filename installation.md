# Installing the Library

If you haven't installed OpenCV yet, please do so by following the instructions below. 

The installation of OpenCV varies between operating systems, so below I am providing instructions for **Windows, Mac, and Linux:**

## Installing OpenCV on Windows

1. Open the command line and type:

``` pip install opencv-python ``` 

2. Then open a Python session and try:

``` import cv2 ```

3. If you get no errors, then OpenCV has been successfully installed and you can skip the next steps. 

4. If there is an error (usually saying that DLL load failed) then please download a precompiled wheel (.whl) file from this link and install it with pip. Make sure you download the correct file for your Windows version and your Python version. For example, for Python 3.6 on Windows 64-bit you would do this:

``` pip install opencv_python‑3.2.0‑cp36‑cp36m‑win_amd64.whl ``` 

5. Then try to import cv2 in Python again. If there's still an error, then please type the following again in the command line:

``` pip install opencv-python ``` 

6. Now you should successfully import cv2 in Python.

## Installing OpenCV on Mac

Currently some functionalities of OpenCV are not supported for Python 3 on Mac OS, so it's best to install OpenCV for Python 2 and use Python 2 to run the programs that contains cv2 code. Its' worth mentioning that Python 2 is installed by default on Mac, so no need to install Python 2. Here are the steps to correctly install OpenCV:

1. Install brew:

Open your terminal and paste the following:

```/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"```

2. OpenCV depends on GTK+, so please install that dependency first with brew (always from the terminal):

``` brew install gtk+ ``` 

3. Install OpenCV with brew:

```brew install opencv ```

4. Open Python 2 by typing:

``` python ```

5. Import cv2 in Python:

```import cv2  ```

If you get no errors, that means OpenCV has been successfully installed.

## Installing OpenCV on Linux

1. Please open your terminal and execute the following commands one by one:

```
    sudo apt-get install libqt4-dev
    cmake -D WITH_QT=ON ..
    make
    sudo make install
```
2. If that doesn't work, please execute this:
```
   sudo apt-get install libopencv-* 
```
3. Then install OpenCV with pip:

```pip install opencv-python ```

4. Import cv2 in Python. If there are no errors, OpenCV has been successfully installed.



