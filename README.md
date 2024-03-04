# fragrant
This is the code about the paper:``FRAGRANT: Frequency-Auxiliary Guided Relational Attention Network for Low-Light Action Recognition.''

Video action recognition aims to classify actions within sequences of video frames, which has important applications in computer vision fields. Existing methods have shown proficiency in well-lit environments but experience a drop in efficiency under low-light conditions. This decline is due to the challenge of extracting relevant information from dark, noisy images. Furthermore, simply introducing enhancement networks as preprocessing will increase the handling burden of the video. To address this dilemma, this paper presents a novel frequency-based method, FRequency-Auxiliary Guided Relational Attention NeTwork (FRAGRANT), designed specifically for low-light action recognition. Its distinctive features can be summarized as 1) A novel Frequency-Auxiliary Module (FAM) that focuses on informative object regions, characterizing action and motion while effectively suppressing noise. 2) A sophisticated Relational Attention Module (RAM) that enhances motion representation by modeling the local s between position neighbors, thereby more efficiently resolving issues such as fuzzy boundaries. Comprehensive testing demonstrates that FRAGRANT outperforms existing methods, achieving state-of-the-art results on various standard low-light action recognition benchmarks. Main components are frequency introducing and relational attention module.

![image](https://github.com/lwxfight/TEST/blob/master/Fragrant/figures/framework.png)

# Instructions(V1)

# Requirements and Dependencies

We assume the environment and the training phase for our experiments following the uniformer: https://github.com/Sense-X/UniFormer

Notably, the sota backbone can be change to UniFormerV2 and others.



# Note

\#Frame = \#input_frame x \#crop x \#clip

- \#input_frame means how many frames are input for model per inference
- \#crop means spatial crops (e.g., 3 for left/right/center)
- \#clip means temporal clips (e.g., 4 means repeated sampling four clips with different start indices) 



# InFAR Dataset：

## Training：

1. Download the source videos from [InFAR](https://pdf.sciencedirectassets.com/271597/1-s2.0-S0925231216X00355/1-s2.0-S0925231216307044/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGUaCXVzLWVhc3QtMSJGMEQCIHD04Oozkk5poSSzJvQBiTa9CPNHbx5gnck%2FK5YYWKbGAiAa4nYPQuHt7eGSJA4X7Eyab6YtEdw%2FuOGf3nzlz14V%2FiqzBQhdEAUaDDA1OTAwMzU0Njg2NSIMLgANyHQugA6cyh8HKpAFPakk3qDQsADSHXUEwzexRUrhEmwI1u%2BbIGkkEBLNX5O1MHkXDu8HlCGmQJd6SvAeB%2FK0hANHOAgzlRx8G%2Bf3OM0Wz%2FgpL0DsQmGJkl0z21el0ZcdUXnOSViqNccrs8Jm%2BKjMCKD499Zt3t8%2FKDQt1W5J7mSRZwkSK3kHu6CatJUlpqsjmTYy4pSNl8FRFxLWMNSE227GhBjoZtw1NbJqbSGddb%2FYtfDqczR3sCtYkG7Yd6qBWIdMA2d2OPN1PDDZKMbwmAUXE1ukqi%2FX19p3InYvSMUem8uQu8uuq68BIDr%2F%2BruuF1NrgmZrosJQdo%2BayEfXcs2QzmIMJanRxs1a5dSPNIt%2BVc6LNQIbkBWKn%2FilfPFCv31Rq4v06iYNG2YIuWKdM3nQPaWiup9Y17QPILQUMePZ9o6Pqs98y3apob063gZ%2B9kx6S1xQev%2Ffv3M%2Fq0oDAjM7ISoIqqTS8G0Out5kodWPbArDxcjibsL6wlUUdwcGaMT0CIjeMhCFG62%2FF%2FGuYm2VoVQYGGA2rN0QLIN1fOKa9OBSqRe6pPOqr4KAkM7xck%2BC2OKPPgvN%2Fd16qGW36HXLmBMSLedkcbJThM6cIgsQ6jtfVuRvMZHT%2B0gJoNqGIVmK3DMCc3royLFrhYiWZR7dhZomxYhZ%2FWfsh8mIW5rESc%2B9SAp2lQr0B87TS9l%2Bb%2FMviBhzFsdQrAe0BxMeNfGdzeHU6bI2z4No5LZlC71RLA1fyy3Erb%2B2sD%2BlN%2Fj%2Bp6cfwnzDSuK5vNrut6p1utI8VANjQuyOkFeGd3eHy78df%2Fgs9DNSak6iy1CYvjxTiTR%2FESWyHultubj4tgQjcmr3RuLqgnxeZ2p2o62WVulDbP19asRlVdjDZz4wptGRrwY6sgEO02m%2BDXhlMQ3wM8r6757v9dKxc1YzH5DlsTg7dKmXldp36j%2BcRCK9chXMPPESB0B3HFgfqFZcV6b6rY1PXo%2BCChlDT6e39lgtKPrAGnpCGYzVipM9lEUmc5QqRZMkUH6PTW5Vr25VHGxgRgWRx%2B9I4ZGPmKn4tuD9lKZTadurWsJ5FvnSzx1NvyWzoF7aheIMYXkfgbLjj5U7Qayn68fFpzVGwybM8%2FkMlEdBoYYL8Vvi&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240303T124336Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTE3ZXTVZ%2F20240303%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=98598933d0746d50b1c5f2d3f92ef92d647f25278a0225ded8ea9e3e25821715&hash=369e2f780d9e2172790af4a12b9a99674416a171fbd816729a00210cc681b517&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925231216307044&tid=spdf-a3aea41e-d10e-470f-ae69-67361e819c80&sid=3a20627b30a4074cf2299a15856c0831fb4dgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0e155a510a5901555f56&rr=85e9b691db58f23c&cc=tw) as groundtruth.

2. Training

  coming soon

## Testing:

Specifically, we need to create our new config for testing and run multi-crop/multi-clip test:

1. Copy the training config file `config.yaml` and create new testing config `test.yaml`.

2. Change the hyperparameters of data (in `test.yaml`):

   ```yaml
   DATA:
     TRAIN_JITTER_SCALES: [224, 224]
     TEST_CROP_SIZE: 224
   ```

3. Set the number of crops and clips (in `test.yaml`):

   **Multi-clip testing for Kinetics**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 4
   TEST.NUM_SPATIAL_CROPS 1
   ```

   **Multi-crop testing for Something-Something**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 1
   TEST.NUM_SPATIAL_CROPS 3
   ```

4. You can also set the checkpoint path via:

   ```shell
   TEST.CHECKPOINT_FILE_PATH your_model_path
   ```

5. An example:

```
python tools/run_net.py --cfg [your file path].infar_config.yaml DATA.PATH_TO_DATA_DIR /data/datasets/infrared_action_dataset   DATA.PATH_LABEL_SEPARATOR ","   TRAIN.EVAL_PERIOD 5   TRAIN.CHECKPOINT_PERIOD 1   TRAIN.BATCH_SIZE 128   NUM_GPUS 1  UNIFORMER.DROP_DEPTH_RATE 0.1   SOLVER.MAX_EPOCH 100   SOLVER.BASE_LR 4e-4   SOLVER.WARMUP_EPOCHS 10.0  DATA.TEST_CROP_SIZE 224   TEST.NUM_ENSEMBLE_VIEWS 4   TEST.NUM_SPATIAL_CROPS 1   TRAIN.ENABLE False   TEST.CHECKPOINT_FILE_PATH []   RNG_SEED 6666   OUTPUT_DIR  []
```



# Your Own Dataset：

1. Prepare data
2. Training
3. Coming soon
