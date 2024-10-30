# Nonlinear-Channel-Capacity
To run the code
* conda create --name myenv python=3.11 pytorch 
* conda activate myenv
* pip install -r requirements.txt


The configuration files are inside "args/"

---Main Loops---
* In order to run Treat Interference as Noise -> interference.py
  *  Configuration file name -> arguments-interferencei.yml (i =[1,11]) (done for remote jobs)
* Without interference -> main.py
  * Configuration file name -> arguments.yml
* TDM -> main.py --config=arguments-TDM.yml

---Results---
* Res -> No interference
* Res-Int -> Treat Interference as Noise
* Res-TDM -> TDM


---Files---
* blahut_arimoto_capacity -> for no interference case, the mutual information maximization with respect to probability distribution is ran with BA
* bounds.py -> information theoratical bounds for no interference nonlinear channels
* Different Regime Mutual Information Calculations:
  * First Regime-> \phi(X) + Z, where Z is the noise, \phi is the nonlinearity
  * Second Regime-> \phi(X+Z)
  * Third Regime-> \phi(X+ Z_1) +Z_2
* nonlinearity_utils-> functions related to nonlinearities
* utils -> general functions


