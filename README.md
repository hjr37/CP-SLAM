<!--   CP-SLAM   -->
<p align="center">
  <a href="">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cp-slam/images/logo-1.jpg" alt="Logo" width="75%">
  </a>
</p>
<p align="center">
  <h1 align="center">CP-SLAM: Collaborative Neural Point-based SLAM System[NeurIPS'23]</h1>
  <p align="center">
    <a href="https://github.com/hjr37/"><strong>Jiarui Hu</strong><sup>1</sup></a>
    ·
    <a><strong>MaoMao</strong><sup>1</sup></a>
    <a href="http://www.cad.zju.edu.cn/home/bao/"><strong>Hujun Bao</strong><sup>1</sup></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/gfzhang/"><strong>Guofeng Zhang</strong><sup>1</sup></a>
    ·
    <a href="https://zhpcui.github.io/"><strong>Zhaopeng Cui</strong><sup>1*</sup></a>
    <br>
    <sup>1 </sup>State Key Lab of CAD&CG, Zhejiang University<br>
    <sup>* </sup>Corresponding author.<br>
  </p>
  <h3 align="center"><a href="https://zju3dv.github.io/cp-slam/">🌐Project page</a> | <a href="https://arxiv.org/abs/2311.08013">📝Paper</a> | <a href="https://www.youtube.com/watch?v=Vbubr-3LH_A">📽️Video</a></h3>
  <div align="center"></div>
</p>

<p align="left">
  <p style="text-align: justify;">This is the official implementation of <strong>CP-SLAM: Collaborative Neural Point-based SLAM System</strong>. CP-SLAM system demonstrates remarkable capabilities in multi-agent deployment and achieves state-of-the-art performance in tracking, map construction, and rendering.</p>
  <a href="">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cp-slam/images/pipeline.jpg" alt="CP-SLAM pipeline" width="100%">
  </a>
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cp-slam/video/single.gif" alt="Single GIF" width="48%">
  <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cp-slam/video/collaboration.gif" alt="Collaboration GIF" width="48%">
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 15px 15px 5px; border-style: solid; border-width: 1px;'>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#News">News</a>
    </li>
    <li>
      <a href="#Dataset Download Link">Dataset Download Link</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li><a href="#run">Run</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ol>
    </li>
    <li>
      <a href="#Acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#Citation">Citation</a>
    </li>
  </ol>
</details>


# News

- [x] 2024.04.11 --- We have updated the `README.md` and are preparing to open-source our code！  
- [x] 2024.04.26 --- Code for some functional modules, including `loop detection`, `pose graph`, `federated center`, and `shared data structure` (detailed comments will come soon).
- [x] 2024.05.08 --- Code for main parts, including `optimizer`, `renderer`, `fusion center`, and `tracking and mapping modules`.
- [ ] Installation setup

# Dataset Download Link
<p style="text-align: justify;">
We provide the <a href="https://huggingface.co/datasets/wssy37/CP-SLAM_dataset">Download link</a> to

- Four single-agent trajectories. Each contains 1500 RGB-D frames.
- Four two-agent trajectories. Each  is divided into 2 portions, holding 2500 frames, with the exception of Office-0 which includes 1950 frames per part.
- Two pre-trained NetVLAD models for the loop detection module. 
</p>

# Installation

- ### Method 1 step-by-step set up(Recommended)


- ### Method 2 Configure the environment in one line



# Usage

## Run

## Evaluation

# Acknowledgement

# Citation
```
@misc{hu2023cpslam,
      title={CP-SLAM: Collaborative Neural Point-based SLAM System}, 
      author={Jiarui Hu and Mao Mao and Hujun Bao and Guofeng Zhang and Zhaopeng Cui},
      year={2023},
      eprint={2311.08013},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
