# Starting point

### **Task summary**

You will take as input cropped images of real photoage from our car. Based on those images you’ll train a model to predict the steering wheel angle. You will have 3 types of models where all take same input but with different output. Details in [the challenge page](https://adl.cs.ut.ee/teaching/the-rally-estonia-challenge) . 

## Getting HPC access

The required data and processing power needed for training is large, so using HPC is inevitable. 

If you have access to HPC please go to **step 4 of Instructions**. If you don’t,  you can gain access to HPC by filling [this forum](https://hpc.ut.ee/getting-started/access/HPC-services). 

### Filling Instructions

1. Choose “Rocket (HPC)” for computing service, and “Rally Estonia Challenge” under project name.
2. For account information, fill user name and email, Also fill the University of Tartu username and institute.
3. For billing contact,
    1. If you are taking this project under a course, fill the course responsible name and email.
    2. If you are taking this project outside of a course,
        1. Name: Tambet Matiisen
        2. Email: [tambet.matiisen@ut.ee](mailto:tambet.matiisen@ut.ee)
4. Once your request approved, contact one of the organizers to access to the compitition files.

Afterwards, you can access the HPC by entering the following command line to Terminal (macOS, Linux), for systems that doesn’t support shell, you might find [puTTY](https://www.putty.org/) or [Termius](https://termius.com/) useful.

```bash
ssh username@rocket.hpc.ut.ee
```

- When in doubt about any command just use
    
    ```bash
    man the_command
    ```
    

## Slurm

Slurm is a workload manager to assure healthy competition over resources. This means, you need to ask for the resources that you need to complete your job. Your request will enter the queue and you’ll get the resources you asked for. Pay attention that the queue is not only about First In First Served, but also about how much resources you are asking for.

### Loading resources

In slurm, to avoid everyone downloading same things and waste space, there are an existing library that each user can load. 

For example, you can load “conda” or “ffmpeg” without installing them yourself.

```bash
module load ffmpeg
module load any/python/3.8.3-conda
```

### Work sessions

There are two ways to launch task on the cluster: 

- **Interactive session**
    - This is to be used when live monitoring or developing/debuging the training process or even prepare your environment.
    - Below is an example:
        
        ```bash
        srun --partition=gpu --gres=gpu:tesla:1 -A "bolt" --time=02:00:00 --pty bash
        ```
        
    - After the resources are allocated, you’ll be able to proceed.
- **Processing sessions**
    - To be used when launching training or evaluation
    
    ```bash
    #!/bin/bash
    #SBATCH -A "bolt"
    #SBATCH -J ADL
    #SBATCH --partition=main
    #SBATCH -t 8-
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=5000
    
    module load ffmpeg
    module load any/python/3.8.3-conda
    python .....
    ```
    

### Hints

```bash
#List your jobs, jobs_id, and their status
squeue -u username

#Cancel a job you launched  
scancel job_id

#Check the GPU 
nvidia-smi

#Check the status of the cluster nodes (try to launch on "idle" partitions)
sinfo

```

### Avoid

- Avoid launching  tasks on the login nodes
- Never “ssh” to nodes and launch your code there, this can get your **account suspended**.

## Preparing Miniconda/anaconda on HPC

There are many packages that need to be installed, and different tasks might need confilicting packages. It is always recommended to use Anaconda.

Anaconda is package management services that allows you to create different environments while keeping each environment packages seperated from the other.

You can start using anaconda (here conda as the light version of anaconda) by  

```bash
module load any/python/3.8.3-conda
```

Afterward you can work with conda commands right away.  You can read more about [conda here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) .

- ************************************Installing mamba:************************************ Due to large dependencies between packages, the conda default packages resolver might take long time. Installing mamba will decrease the resolving time significantly. To install mamba :
    
    ```bash
    conda install mamba -n base -c conda-forge
    ```
    

Info regarding [mamba here](https://anaconda.org/conda-forge/mamba). Note that after installing you can use the command “`mamba`” instead of “`conda`” if you want to use the mamba resolver.

## Training

- Create the training environment. (This might take time,  10 - 60 minutes)
    
    ```bash
    #Clone this repository
    git clone https://github.com/UT-ADL/e2e-rally-estonia.git
    cd e2e-rally-estonia
    
    #if mamba installed
    mamba env create -f e2e.yml
    #else 
    conda env create -f e2e.yml
    
    #Activate the environment
    conda activate e2e
    ```
    
- [Optional] Initialize weights and biases library `wandb`. It is highly recommended as it ease up tracking your experiments.
    - Create an account at [wandb.ai](http://wandb.ai) then login
    
    ```bash
    wandb login
    ```
    
    - More information about `wandb`  [here](https://docs.wandb.ai/guide)
- Check you are using the GPU ([source](https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu)):
    
    ```bash
    python
    >>> import torch
    >>> torch.cuda.is_available()
    True
    >>> torch.cuda.current_device()
    0
    >>> torch.cuda.get_device_name(0)
    'Tesla V100-PCIE-32GB'
    ```
    
- To check everything is working for you, execute the following code, **(make sure you booked  - 1 GPU, 16 CPU, and 16 GB memory on Slurm )**

```bash
python train.py --input-modality nvidia-camera --output-modality steering_angle --patience 10 --max-epochs 1 --model-name steering-angle --model-type pilotnet-conditional --dataset-folder /gpfs/space/project/rally2023/rally-estonia-cropped-antialias --wandb-project testing
```

- **Hints**
    - As the training data is large and it needs pre-processing, the code contain workers that fetch and preprocess the data before being used by the network. The number of workers can be specified when you start the training by `--num-workers 16` . The first time you start training, use a very small number like 2, after than the code will suggest a propriate number of the workers based on the booked resources. In general, each worker need 1 cpu core and 1GB of memory, so if you asked slurm for 16 cpus and 16GB of memories you can launch 16 workers.
    - Use the default value for number of workers `--num-workers 16` . Based on my tests, it was the fastest.
    - Using `rsync` is not essential (save 1 minute per epoch) but It can be usefule at time of network conjestion. **Only use the below command to** prevent doubling the space if multiple teams used different names.
    
    ```bash
     rsync /gpfs/space/projects/tmp/rally-estonia-cropped-antialias /tmp/    
    ```
    

## Models evaluation

We use a simulation to evaluate the models. Which means we can test the trained models fast and safely. To setup the evaluation repository you can follow the guide presented in the below repository. **Please note** that when preparing the environment for the HPC, you don’t need to install system packages “sudo apt install …” 

[Vista-evaluation](https://github.com/UT-ADL/vista-evaluation/)

### Model adaptation
Onnx models trained using e2e-rally-estonia includes the batch size in the input/output of the model, which raises an error during evaluation. Adapting the following code to your model would resolve the error:

```python
import onnx
model = onnx.load("model.onnx")
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value =1
onnx.save("model_modified.onnx")
```
