# DeepGPET: fully automated choroid region segmentation in OCT

## DeepGPET

DeepGPET is a fully automatic, deep learning based method for choroid region segmentation in optical coherence tomography (OCT) scans. This algorithm can be used to segment the choroid region, and then subsequently measure clinically relevant metrics such as choroid thickness, area and volume. The method was trained on labels from a previously published, semi-automatic algorithm [GPET](https://ieeexplore.ieee.org/document/9623378).

![schematic](install/figures/schematic.png)

The description of the model can be found [here](https://tvst.arvojournals.org/article.aspx?articleid=2778573). The paper for this methodology has been published in Translations Vision Science & Technology, Association for Research and Vision in Ophthalmology (ARVO).

---

### Project Stucture

```
.
├── choseg/         # core module for carrying out choroid region segmentation and derived regional measurements
├── notebooks/		# Jupyter notebooks which stores example data and a demo of the pixel GUI to select the fovea
├── install/		# Files to help with installation
├── .gitignore
├── README.md
└── usage.ipynb		# Anaconda prompt commands for building conda environment with core packages
```

- The code found in `choseg`
```
.
├── choseg/                             
├───── metrics/         # Code to calculate downstream regional measures such choroid thickness, area and (subregional) volume
├───── __init__.py
├───── inference.py     # Inference classes for segmenting
└───── utils.py         # Helper functions for plotting and processing segmentations
```

- The code found in `choseg/metrics`
```
.
├── metrics/
├───── __init__.py                          
├───── choroid_metrics.py   # Code to calculate choroid measurements such choroid thickness, area and (subregional) volume
├───── choroid_utils.py     # Utility functions for measuring the choroid
└───── pixel_gui.py         # OpenCV-based implementation of a simple GUI to select pixels manually
```

- The files found in `install`
```
.
├── install/                             
├───── figures/             # Folder of images for README file.
├───── conda.pth            # File to link DeepGPET's local repository folder to the conda environment
└───── install.txt          # Anaconda Prompt commands to built conda environment and packages
```

---

## Getting started

To get a local copy up, follow the instructions below.

1. You will need a local installation of Python to run DeepGPET. We recommend a lightweight package management system such as Miniconda. Follow the instructions [here](https://docs.anaconda.com/free/miniconda/miniconda-install/) to download Miniconda for your desired operating system.

2. After downloading, navigate and open the Anaconda Prompt and clone the DeepGPET repository.

```
git clone https://github.com/jaburke166/deepgpet.git
```

3. Create environment and install dependencies to create your own environment in Miniconda. Open Anaconda Prompt and copy these lines in turn onto the terminal.

```
conda create -n deepgpet python=3.11 -y
conda activate deepgpet
pip install -r requirements.txt
```

4. (Optional) Copy the file `conda.pth` in `install/` file into your python environments `site-packages` directory, commonly found at `path\to\miniconda3\envs\choroid-analysis\Lib\site-packages` on Windows. Make sure to change the file paths in `conda.pth` to the absolute file path that link to the DeepGPET Github repository.
    - This means your conda environment `choroid-analysis` now has access to the files in your local copy of DeepGPET.
    - Importantly, you can now use `import choseg` in any notebook or python script which uses this conda environment.
    - 

5. Activate your new environment using `conda activate deepgpet` on an Anaconda Prompt terminal.
 
6. Launch notebooks using `jupyter notebook` or jupyter labs using `jupyter lab` and see the minimal example below so that you can analyse your own choroids!

Done! You have successfully set up the software to analyse the choroid region in optical coherence tomography B-scan data!

**note**: If you have any problems using this method, please do not hesitate to contact us (see contact details at the end of this README)!


### Minimal example

```
from choseg import inference, utils
from choseg.metrics import choroid_metrics

# Load model, default threshold of 0.5
deepgpet = inference.DeepGPET()

# Load image
img_path = r"notebooks\example_data\image1.png" #"path\to\img"
img = utils.load_img(img_path)

# Segment
img_seg = deepgpet(img)
img_seg_cmap = utils.generate_imgmask(img_seg,0.5,0) # Creates an RGBA colourmap to overlay segmentation onto image
img_trace = utils.get_trace(img_seg,0.5) # Detects the upper and lower choroid boundaries based on the mask
utils.plot_img(img, img_trace, cmap=img_seg_cmap, sidebyside=True) # Plots the image, trace and segmentation colour map

# Measure choroid thickness and area at default 3mm region of interest
choroid_metrics.compute_choroid_measurement(img_seg)
# thickness (temporal, subfoveal, nasal) (190, 290, 130), choroid area 1.374898mm2
```

Please refer to `usage.ipynb` for a more in depth description of segmenting the choroid using DeepGPET, and measuring choroid thickness and area.

---

### Related repositories

If you are interested in choroid analysis in OCT images, check these repositories out:

* [OCTolyzer](https://github.com/jaburke166/OCTolyzer): A fully automatic analysis toolkit for segmentation and feature extracting in OCT data (and scanning laser ophthalmoscopy data, or SLO). 
* [Choroidalyzer](https://github.com/justinengelmann/Choroidalyzer): A fully automatic, deep learning-based tool for choroid region and vessel segmentation, and fovea detection in OCT B-scans.
* [MMCQ](https://github.com/jaburke166/mmcq): A semi-automatic algorithm for choroid vessel segmentation in OCT B-scans based on multi-scale quantisation, histogram equalisation and pixel clustering.
* [EyePy](https://github.com/MedVisBonn/eyepy): A selection of python-based readers of various file formats of OCT B-scans, including `.vol` and `.e2e` format from Heidelberg.

If you are interested in en face analysis of the retina using colour fundus photography (CFP) or scanning laser ophthalmoscopy (SLO), check these repositories out:
* [AutoMorphalyzer](https://github.com/jaburke166/AutoMorphalyzer): A fully automatic analysis toolkit for segmentation and feature extracting in CFP data, a restructured version of [AutoMorph](https://github.com/rmaphoh/AutoMorph).
* [SLOctolyzer](https://github.com/jaburke166/SLOctolyzer): Analysis toolkit for automatic segmentation and measurement of retinal vessels on scanning laser ophthalmoscopy (SLO) images

---
## Contributors and Citing

The contributors to this method and codebase are:

* Jamie Burke (Jamie.Burke@ed.ac.uk)
* Justin Engelmann (Justin.Engelmann@ed.ac.uk)

If you wish to use this methodology please consider citing our work using the following BibText

```
@article{burke2023open,
  title={An Open-Source Deep Learning Algorithm for Efficient and Fully Automatic Analysis of the Choroid in Optical Coherence Tomography},
  author={Burke, Jamie and Engelmann, Justin and Hamid, Charlene and Reid-Schachter, Megan and Pearson, Tom and Pugh, Dan and Dhaun, Neeraj and Storkey, Amos and King, Stuart and MacGillivray, Tom J and others},
  journal={Translational Vision Science \& Technology},
  volume={12},
  number={11},
  pages={27--27},
  year={2023},
  publisher={The Association for Research in Vision and Ophthalmology}
}
  ```

 
 
 
 
 
 
 
 
