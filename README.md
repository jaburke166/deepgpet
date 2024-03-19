# DeepGPET: fully automated choroid region segmentation in OCT

Repository storing fully automatic DL-based approach, DeepGPET, for choroid region segmentation in optical coherence tomography images.

The description of the model can be found [here](https://tvst.arvojournals.org/article.aspx?articleid=2778573). The paper for this methodology has been published in Translations Vision Science & Technology, Association for Research and Vision in Ophthalmology (ARVO).

---

### Project Stucture

```
.
├── choseg/         # core module for carrying out choroid region segmentation and derived regional measurements
├── example_data/	# example OCT B-scans to demonstrate usage
├── notebooks/		# Jupyter notebooks to demonstrate usage
├── .gitignore
├── README.md
├── LICENSE
└── install.txt		# Anaconda prompt commands for building conda environment with core packages
```

- The code found in `choseg`
```
.
├── choseg/                             
├───── metrics/         # Code to calculate downstream regional measures such choroid thickness, area and (subregional) volume
├───── weights/		    # Stores the model weights.
├───── __init__.py
├───── inference.py     # Inference classes for segmenting
└───── utils.py         # Helper functions for plotting and processing segmentations.
```

---

## Getting Started

To get a local copy up and running follow these steps.

1. Clone the DeepGPET repository via `git clone https://github.com/jaburke166/deepgpet.git`.

2. Follow the instructions [here](https://docs.anaconda.com/free/miniconda/miniconda-install/) to download Miniconda3 for your desired operating system.

3. Copy the commands into your anaconda prompt found in `install.txt` to create your own environment in Miniconda.
    - Note if you have a GPU running locally to use DeepGPET, line 2 should read `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  
4. Copy the `conda.pth` file into your python environments `site-packages` directory, commonly found at `path\to\miniconda3\envs\choroid-analysis\Lib\site-packages`
    - Change the file paths in `conda.pth` the absolute file paths that link to the DeepGPET Github repository.
    - This means your conda environment `choroid-analysis` now has access to the files in the DeepGPET.
  
5. Launch notebooks using `jupyter notebook` or jupyter labs using `jupyter lab`

Done! You have successfully set up the software to analyse the choroid in OCT B-scans!

### Minimal example

```
from choseg import inference, utils

# Load model
model_path = r"choseg\weights\model_weights.pth"
deepgpet = inference.InferenceModel(model_path=model_path, threshold=0.5)

# Load image
img_path = r"notebooks\example_data\image1.png" #"path\to\img"
img = utils.load_img(img_path)

# Segment
img_seg = deepgpet(img)
utils.plot_img(img, cmap=utils.generate_imgmask(img_seg), sidebyside=True)
```

---
## Contributors and Citing

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

 
 
 
 
 
 
 
 
