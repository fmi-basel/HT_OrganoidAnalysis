# HT_OrganoidAnalysis

This repository contains the code used for high-throughput image analysis in the manuscript "Dynamics and plasticity of stem cells in the regenerating human colonic epithelium" by Oost and Kahnwald et al. from the Liberali Lab.

## Installation

To set up the virtual environment and install the required dependencies, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/fmi-basel/HT_OrganoidAnalysis.git
  
2. Navigate to the project directory:

   ```bash
   cd HT_OrganoidAnalysis
  
3. Create a virtual environment (Python 3.11.3 is recommended):

   ```bash
   python -m venv venv
   
4. Activate the virtual environment:

- Windows:

   ```bash
   .\venv\Scripts\activate

- macOS/Linux:

   ```bash
   source venv/bin/activate

5. Change directory to the downloaded repository and install the required packages using pip:

   ```bash
   pip install -r requirements.txt

You are now ready to run the code! Be sure to activate the virtual environment before running any scripts.


## Notebooks Overview

### 0_Segmentation
This notebook employs CellPose to segment objects from maximum-intensity-projections (MIPs). MIPs should be stored in a designated folder, and a unique naming convention for the segmentation channel can be specified. An example CellPose model for mature organoids is available in this repository. After segmentation, objects must be extracted from the overview MIP with a bounding box based on their segmentation mask into separate folders with a specific folder structure. See below. 

### 1_FeatureExtraction
Extracts 2D organoid-level morphological (including crypt-associated) and intensity features from objects, saving the results in an AnnData object and a .csv file. The AnnData object is used for further processing in subsequent scripts.

**Folder Structure Expectation:**

- ExperimentFolder with PlateSetup_BARCODE.xls files. See section **Plate Setup file**
  - Well folder (e.g., B04/B05/B06, etc.)
    - Object folder (must be in format object_ID. For example, object_1, object_2, etc.)
      - Files for one extracted object according to these rules:
        1. Channels are indicated with a unique identifier in their name: Z01C01 (channel 01), Z01C02 (channel 2), Z01C03 (channel 3), Z01C04 (channel 4). Currently, up to four channels are supported.
        2. Images of segmented channels end with *_TIF-OVR-MIP.tif, while the image of the segmentation mask ends with *_MASK.tif.
        3. Every object folder has **at least** one channel image (e.g., Z01C01_TIF-OVR-MIP.tif) and **exactly** one segmentation mask for this object (e.g., Z01C01_MASK.tif).
       
### 2_Filtering
This script filters debris as wrongly-segmented objects using a subset of extracted features with visual feedback.

### 3_QualityControl
This script tests for the normal distribution of features. Area-associated features are log1p transformed. Additionally, it checks for imaging plate bias.

### 4_Trajectory
Uses a user-defined subset of features for Slingshot trajectory inference.


The `Functions.py` file houses the main functions utilized in other scripts. It is recommended to keep it in the same folder as the jupyter notebooks.


## Plate Setup

To link wells to a specific condition, the ExperimentFolder must contain a PlateSetup_BARCODE.xls (in exactly this naming convention) file for every imaging plate to be analysed. The barcode herein must be the same as specified in the notebook 1_FeatureExtaction. Setup files are available as templates in a 96 and 384 well plate format in this repository and contain four tabs:

1. **CellLine**: Identifier for used cell/organoid line in each specific well of the respective imaging plate.
2. **Antibodies**: Identifier for the used antibody-mix in each specific well of the respective imaging plate. Contents of the antibody mix must be specified in 1_FeatureExtraction to allow linking of features to names of stainings.
3. **Medium**: Identifier for the used medium in each specific well of the respetive imaging plate.
4. **Other**: Additional identifier for a used condition, such as fixation timepoint.

All wells which have not been imaged should remain empty!
