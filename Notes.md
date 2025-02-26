## Coding Notes
* Regime 2 has no updates from the last changes - it's not an interesting regime
* GD over learning rate, alphabet size functions are canceled - Future work-- they were not working properly
* First Regime and Third Regime- old capacity calculations are commented out- Not necessary now, current calculations are simpler and seems to be working fine


## February 23rd
* ADC results in higher capacity?? Power mapping is smart
*  Haven't checked complex in main.py - currently does not work, pretty sure
* Interference checking
* And currently ADC or Interference - I aim to make it work in the First Regime

## February 24th
* ADC is working - on alphabet_y
* main is running with complex
* move the bounds out and try mmse!! (You are here)

## February 25th
* MMSE is working for the first regime (Both complex and real) with ADC
* Lied... real is a problem, weird -- Trying to figure this out
* MMSE is working but for no ADC case, it gives nan for high power values ? Dont get this. Because sigma_y_2 gets negative and it should be the case (More points fixed this- 2000 Samples currently)
* Lower bound Tarokh seems to not be working with hardware params-consider to check
-- TO DO --
* I need to focus on Interference 
* Also, plot different bits results together (One of them is running in HPC) and max capacity results from learned with different power consumptions of Hardware (plot_hardware_res.py)
