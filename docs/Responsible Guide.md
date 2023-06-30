# Responsible Technical guide

This technical guide provides information on data pipline for developing autonomous driving competition. From collecting the data to car testing.

Please visit [students guide](https://github.com/UT-ADL/e2e-rally-estonia/blob/master/docs/Start_point.md) first.

General information about the data flow:

- The data is recorded from the car sensors through ROS in a bag file (rosbag).
- The bag filed then converted to a trace and training data.
    - Trace: is used in vista-evaluation. So model’s can be evaluated safely.
    - Training data: is an extract of bagfile into images and other sensory csv files. The training data can be a full images, cropped or anti-aliased.

### Creating Traces

Traces conversion is done through [vista-evaluate](https://github.com/UT-ADL/vista-evaluation/) repository.

- **Creating an HPC CPU based  environment to convert traces**
    
    The job of this environment is to convert bag files to vista trace. 
    
    To create this environment: 
    
    ```bash
    conda create -n vista_trace python=3.8
    conda activate vista_trace
    git clone https://github.com/UT-ADL/vista-evaluation.git
    cd vista-evaluation
    pip install -r requirements.txt
    pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag roslz4 cv-bridge
    
    ```
    
- **Start converting**
    
    In the vista-evaluation repository →  create_trace.py there are two names for camera feed topic which you need to adapt manually (currently): 
    
    ```python
    # Old topic name
    CAMERA_TOPIC_30HZ = '/interfacea/link2/image_raw/compresse
    # New topic name
    CAMERA_TOPIC_30HZ = '/interfacea/link2/image/compressed'
    ```
    
    To start the code: 
    
    ```bash
    #Example launch code
    python create_trace.py --bag /gpfs/space/projects/Bolt/bagfiles/2021-05-20-12-36-10_e2e_sulaoja_20_30.bag --resize-mode resize --output-root /gpfs/space/projects/Bolt/end-to-end/vista/challenge_traces/ --force Force
    ```
    
- **Create traces in bulks in slurm**
    
    The batch file has many assumptions (location of vista-evaluation, location of bag files, location and name of list of bags to be converted, the name of conda environment, etc..) read through and modify as fit.
    
    To use the below code, you need to be 1 directory before vista-evaluation folder and you put the bags names in the same directory “paper_bags.out”
    
    ```bash
    #!/bin/bash
    #SBATCH -A "bolt"
    #SBATCH -J ADL
    #SBATCH --partition=main
    #SBATCH -t 8-
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=5000
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vista_trace
    module load ffmpeg
    while read line
    do
            cd vista-evaluation
            if [[ $line != \#* ]]; then
                    # Process the line here
                    python create_trace.py --bag /gpfs/space/projects/Bolt/bagfiles/$line --resize-mode resize --output-root /gpfs/space/projects/Bolt/end-to-end/vista/challenge_traces/ --force Force
            fi
            sleep 1 #7200 # Wait for 2 hours before reading the next line
            echo "Current time: $(date)" # Print the current time
            cd ..
    done < paper_bags.out
    ```
    

### Creating Training data

Creating training data depends on [e2e-rally-estoina](https://github.com/UT-ADL/e2e-rally-estonia) repository. Make sure you git the repo before continuing.

- **Creating environment on the HPC**
    - First, download following file to HPC (you can use scp command for that)
        
        [ros.yml](Responsible%20Technical%20guide%2069beda1f64b94fefa24b3610de06d9a0/ros.yml)
        
    
    ```bash
    conda env create -f ros.yml
    ```
    
- **Converting**
    
    For converting, you need `data_extractor/image_extractor.py` file. use `--help` for arguments information.
    
    ```bash
    conda activate ros
    python image_extractor.py --help 
    python image_extractor.py --bag-file "file" --extract-dir location
    ```
    
    - ************Note************ that data need to be preprocessed [afterward](https://github.com/UT-ADL/e2e-rally-estonia/blob/master/dataloading/preprocess.py).
- **Converting in bulks**
    
    Please check the `*.sh` and `*.job` files in same directory 
    
- **The anti-aliasing dataset**
    
    The antialiasing dataset is generated with [this repository](https://github.com/mbalesni/ebm-driving/blob/master/data_extract/bag_extractor.py)
    
    - mikita generated the data set by `torchvision.transformers.functional` for generating the antialiasing dataset.
    - Our best solution would be to look into how torchvision is doing it and replicat the algorithm.

### Evaluating  models

- To evaluate models look [here](https://github.com/UT-ADL/vista-evaluation/blob/master/evaluation-server/README.md)

### Pre-car preparations (**E2E_nvidia)**

This setup to be locally installed without using the car. To test everything would work and get comfortable what to expect when working with the car.

- The main link for configuration and installation packages exists [here](https://gitlab.cs.ut.ee/autonomous-driving-lab/autoware_mini/) (nested links, which starts [here](http://wiki.ros.org/noetic/Installation/Ubuntu))
- Clone [this repository](https://gitlab.cs.ut.ee/autonomous-driving-lab/e2e/e2e_platform) to the src folder
- Rebuild ros nodes to take into account the new e2e_platform.

In case of facing pacmod error please do the following:

```bash
# kvaser-canlib-dev is found in the following ppa
sudo apt-get -qq install software-properties-common
sudo apt-add-repository -y ppa:astuff/kvaser-linux

# Public AStuff repo
sudo apt-get -qq update
sudo apt-get -qq install apt-transport-https
sudo sh -c 'echo "deb [trusted=yes]https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
sudo apt-get -qq update

# rosdep list
sudo sh -c 'echo "yamlhttps://s3.amazonaws.com/autonomoustuff-repo/autonomoustuff-public-'$ROS_DISTRO'.yaml" > /etc/ros/rosdep/sources.list.d/40-autonomoustuff-public-'$ROS_DISTRO'.list'
rosdep update
```

- To test everything you can use where the bag_file needs to be located in `[autoware_mini]/src/e2e-platform/data/bags` (parameters details in `[autoware_mini]/src/e2e-platform/launch/`)

```bash
roslaunch e2e_platform start_bag.launch bag_file:=2021-06-07-14-36-16_e2e_rec_ss6.bag
```

- while `rosluanch` is running, in another terminal running `rqt` shows the rosbag topics published. Notably:
    - plugins → Topics → topic monitor
    - plugins → Visualization → image View
- **Car guide**
    
    Commands: 
    
    - The command used for launching the machine is:
    
    ```bash
    roslaunch e2e_platform start_lexus.launch “Then add parameters” 
    ```
    
    To check all the parameters you can find the in e2e_platform→ launch→ check the wanted launch file. 
    
    The important things to notice are the needed parameters to launch students models.
    
    - The command used to record a bag file for all car topics is (stored in launching command directory) :
    
    ```bash
    rosbag record -a -o bagname
    ```
    
- **Possible competition improvements**
    - Put all the competition information in one place.
    - Set office hours to support the students.
    - Make video session or a practice session to get student up to speed with the code.
    - Automate the students models evaluation.
    - Complete information registration about students as this information may need to be transported to courses responsibles later.
    - Many frames contains sun, glare, water, or dirt. LSTM might be beneifical for students.
- **Soft bugs to fix**
    - Falcon 4 bug. It occure on falcon 4. Need to be replicated then reported to HPC.
        
        ![IMG_0920.png](Responsible%20Technical%20guide%2069beda1f64b94fefa24b3610de06d9a0/IMG_0920.png)
        
    - On Converting bagfiles to traces. Some bagfiles time for turnsignal start before the time for images. Which raise an error when introplating the turn signal between two times at the start. Setting the first time stamp for turn signal identical to first image would resolve the issue.
    - The seconds after crash can only be integer currently. Using a float (0.5) raise an error.
        
        Use a model the predict constant value to see to cause more crashes and trace the error fast
        
    - The onnx file is saved with batchsize included (1 or another number) this need to be modified to be dynamic.
    - Topics fix: there are multiple topics names for same content in the bag files. Matching file is required.
    - Allow to dynamically select of training and validation datasets.