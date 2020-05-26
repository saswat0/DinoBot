# DinoBot
A PyTorch based approach for the popular dinosaur game on Google Chrome

# Background and Motivation
Based as an Adaptation to [DeepMind paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) in 2013) on an introductory [blog](https://blog.paperspace.com/dino-run/) presented in PaperSpace based on Deep RL of QNetwrok

## Initial configurations
Download chromedriver from this [link](http://chromedriver.chromium.org/) and extract it to a directory
Install the required python library commands by issuing
```bash
pip install -r requiremenets.txt```

### Arguments available for train.py
```bash
  --chrome_driver_path  CHROME_DRIVER_PATH  Path of chrome driver
  --checkpoint_path     CHECKPOINT_PATH     Path of Pytorch model path
  --nb_actions NB_ACTIONS   Number of possible actions for bot
  --initial_epsilon INITIAL_EPSILON Starting epsilon value for explorations
  --final_epsilon FINAL_EPSILON Final value for epsilon after exploration
  --gamma GAMMA         Value of gamma for attenuation of rewards in next
                        states.
  --nb_memory NB_MEMORY Number of memory to store previous states and rewards
                        for training.
  --nb_expolre NB_EXPOLRE   Number of times for explorations. After this time the 
                        epsilon is in final_epsilon value and the explorations
                        is in its minumum value.
  --is_debug            A flag for debugging. If enabled an OpenCV window is
                        shown that illustrates the feeded image to the netwrok
  --batch_size BATCH_SIZE
                        Batch size for training.
  --nb_observation NB_OBSERVATION
                        Number of observations before starting training
  --use_cuda            Use cuda if it\'s available
  --exploiting          Enable this to skip training
  --log_frequency LOG_FREQUENCY
                        Frequency of logging every time step.
  --save_frequency SAVE_FREQUENCY
                        Frequency of saving state of the training
  --game_speed GAME_SPEED
                        Speed of the game. Higher speeds call for better CPU/GPU.
  --ratio_of_win RATIO_OF_WIN
                        Ration of usage of win actions in training. It should
                        be between (0,1]. 1 means use all actions and 1e-6
                        means small amount of win actions.
  --desired_fps DESIRED_FPS
                        If you want to reduce processing fps to have
                        constant fps in training and testing time.
```

## Running the Code
* Train from scratch with default values
    * python train.py --chrome_driver_path /path/to/chromedriver
* Train more with failures.
    * python train.py --chrome_driver_path /path/to/chromedriver --ratio_of_win 0.1
* Load the pretrained model for testing purposes
    * python train.py --chrome_driver_path /path/to/chromedriver --checkpoint_path ../weights/freezed_model.pth --exploiting

## Pre-trained weight
Generated as a sample after 400 trials of the game