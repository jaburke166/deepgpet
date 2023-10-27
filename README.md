# DeepGPET: fully automated choroid region segmentation in OCT

Repository storing fully automatic DL-based approach, DeepGPET, for choroid region segmentation in optical coherence tomography images.

The description of the model can be found [here](https://arxiv.org/abs/2307.00904). This paper has been accepted to be published in Translations Vision Science & Technology, published by the Association for Research and Vision in Ophthalmology.

---

### Project Stucture

```
.
├── choseg/             # core module for carrying out choroid region segmentation and derived regional measurements
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
├───── metrics/             # Code to calculate downstream regional measures such choroid thickness, area and (subregional) volume
├───── weights/		    # Stores the model weights.
├───── __init__.py
├───── inference.py         # Inference classes for segmenting
└───── utils.py             # Helper functions for plotting and processing segmentations.
```

---

## Getting Started

To get a local copy up and running follow these steps.

1. Clone the repo via `git clone https://github.com/jaburke166/deepgpet.git`.

2. Copy the commands into your anaconda prompt found in `install.txt`.
    - Note if you have a GPU running locally to use DeepGPET, line 2 should be `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

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
@misc{burke2023opensource,
      title={An open-source deep learning algorithm for efficient and fully-automatic analysis of the choroid in optical coherence tomography}, 
      author={Jamie Burke and Justin Engelmann and Charlene Hamid and Megan Reid-Schachter and Tom Pearson and Dan Pugh and Neeraj Dhaun and Stuart King and Tom MacGillivray and Miguel O. Bernabeu and Amos Storkey and Ian J. C. MacCormick},
      year={2023},
      eprint={2307.00904},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
  ```

 
 
 
 
 
 
 
 
