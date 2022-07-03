# Using Deep Learning Techniques to Optimize Access Demand for MMTC Traffic
Repository containing the codes developed while producing my master's thesis, with the title *Using Deep Learning Techniques to Optimize Access Demand for MMTC Traffic*.
The *.mat* files contain stored data. Specifically, the ones in this repository contain data for the trained Deep Q-Network (DQN) agents number 7501 and 7504, which correspond to the ones with the highest reward and average reward (on a 5-episode window), respectively, according to our 1e4-episode training.
The *.mat* files corresponding to the trained neural networks (NNs) are stored in a Google Drive folder (https://drive.google.com/drive/folders/1m4HQ-DcVsMBgddrx7haMwFdN3dtZfg9c?usp=sharing) because of Github's file size limitations. The link is
All files are written in MATLAB R2021b.

* *comparative_analysis*: this script is the one generating the results and graphics presented in Section 3.5. This script uses other subroutines, functions and stored data which are listed below:
  * *epi_sim_estimated_action*: function that generates measurements collected throughout an extended 4000-RAO episode
  * *single_RAO_loop*: function generating a single execution of the ACB and CBRA protocols
  * *Agent7501.mat*: trained DQN agent number 7501
  * *Agent7504.mat*: trained DQN agent number 7504
  * *trained_net_21_1_1_1_Max_10* [in the Google Drive folder]:
  * *trained_net_21_1_1_1* [in the Google Drive folder]:
  * *trained_net_21_16_1_2_Max_10* [in the Google Drive folder]:
  * *trained_net_21_16_1_2* [in the Google Drive folder]:
