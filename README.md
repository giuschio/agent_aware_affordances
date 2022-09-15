# Learning Agent-Aware Affordances for Closed-Loop Interaction with Articulated Objects
[![alt text](images/real_world_interaction_horizontal.jpg?raw=true)](https://youtu.be/StTqXEQ2l-Y?t=35s)
Real-world experiment of opening an oven in two motions. a) & c): Estimated actionability map where the red cross represents the selected interaction point. b) The first interaction pose becomes unfavorable, therefore an update is triggered. d): Successful task completion after the second interaction.
## Introduction
Interactions with articulated objects are a challenging but important task for mobile robots. To tackle this challenge, we propose a novel closed-loop control pipeline, which integrates manipulation priors from affordance estimation with sampling-based whole-body control. We introduce the concept of agent-aware affordances which fully reflect the agent's capabilities and embodiment and we show that they outperform their state-of-the-art counterparts which are only conditioned on the end-effector geometry. 
Additionally, closed-loop affordance inference is found to allow the agent to divide a task into multiple non-continuous motions and recover from failure and unexpected states. Finally, the pipeline is able to perform long-horizon mobile manipulation tasks, i.e. opening and closing an oven, in the real world with high success rates (opening: 71%, closing: 72%).

## About the paper
Authors: [Giulio Schiavi<sup>*</sup>](https://asl.ethz.ch/the-lab/people/person-detail.MjY0OTk4.TGlzdC8xNTg0LDEyMDExMzk5Mjg=.html) ([github](https://github.com/giuschio), [linkedin](https://www.linkedin.com/in/giulio-schiavi-439174221/)),
[Paula Wulkop<sup>*</sup>](https://asl.ethz.ch/the-lab/people/person-detail.MjA0OTUz.TGlzdC8yMDMwLDEyMDExMzk5Mjg=.html),
[Giuseppe Rizzi](https://mavt.ethz.ch/people/person-detail.MjMyMjQy.TGlzdC81NTksLTE3MDY5NzgwMTc=.html),
[Lionel Ott](http://www.ott.ai/),
[Roland Siegwart](https://asl.ethz.ch/the-lab/people/person-detail.Mjk5ODE=.TGlzdC8yMDI4LDEyMDExMzk5Mjg=.html),
[Jen Jen Chung<sup>1</sup>](http://jenjenchung.github.io/anthropomorphic/),     
from the Autonomous Systems Lab, ETH Zurich, Switzerland.    
<sup>*</sup> Equal contribution.   
<sup>1</sup> Also with the School of ITEE, The University of Queensland, Australia.    

Arxiv Version: [https://arxiv.org/abs/2209.05802](https://arxiv.org/abs/2209.05802)

Project Page: [https://paulawulkop.github.io/agent_aware_affordances](https://paulawulkop.github.io/agent_aware_affordances/)    

Project Video: [https://www.youtube.com/watch?v=A_v5GPFaLwU](https://www.youtube.com/watch?v=A_v5GPFaLwU)

## The Code
The code in this repository is available under an MIT license, and can be used to train and test our pipeline. We additionally provide trained network checkpoints and some demos. Please refer to the [documentation on running the code](src/README.md) for additional details. Please note that in order to run this code a RAISIM installation is required. The installation requires a license, which can be requested free of charge for academic purposes.

## Citations
If you use our code in your research, please cite our paper as:

    @misc{schiavi2022learning,
      title={Learning Agent-Aware Affordances for Closed-Loop Interaction with Articulated Objects},
      author={Giulio Schiavi and Paula Wulkop and Giuseppe Rizzi and Lionel Ott and Roland Siegwart and Jen Jen Chung},
      year={2022},
    }

## Acknowledgements   
This work was inspired by the Where2Act framework and uses some code from their implementation, which is available [here](https://github.com/daerduoCarey/where2act).

## Funding
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101017008 (Harmony).