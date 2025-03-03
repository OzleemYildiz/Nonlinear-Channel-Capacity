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
* Plotted different bits results together  and max capacity results from learned with different power consumptions of Hardware (plot_hardware_res.py). There was a wrong HPC - correct one is running in HPC now. Also trying power consumption graph with 5 bits at the same time
* Running also complex- could be memory error lets see
-- TO DO --
* I need to focus on Interference 
* There are Complex-ADC-PP, Interference-Hardware, Int-HD-ADC Runs
* I seem to be able to not save everything in the results folder - Gotta maximize somehow
* Interference without X2 fixed, I could still plot capacity results for different powers in one figure - More informative
* Get in with Third Regime
* How does the approximation look like?

## February 26th
* PP Power runs are done for 5 bits and the missing file has been added to the results folder
* MMSE with ADC is working - Plotted with different bits
* All the complex runs failed - Memory and stuff, deal with it later
* Marginal&Joint Pdf plotting is completed in plt_hardwares.py
* No ADC, Interference results seem make sense, Gaussian and Learned gives the same results since we are not in saturation


--- TO DO ---
* Linear Approximation for the first regime-Check this
* Interference with X2 fixed is running -Check this
* Interference when X2 is not fixed, how should I make sense the data? I am not sure


## March 1st
* Plotting even when R2 is not fixed for different powers is implemented
-TO DO-
* Interference Linear seems off - Check
* Interference KI seems off check

## March 2nd
# NOTE: Don't run with boundary, mmse boundary needs too many samples and we could do it offline
* Interference in First Regime is working - Alphabet of RX2 was not changed with multiplying factor
* Y = sqrt(Gain)*ktanh(x/k) +Z2, we divide everything by sqrt(Gain) to reduce the alphabet_y size - Z2's variance is divided by Gain 
        * Tested in main.py with hardware params - seems to be working, ADC also is working
* Why was I taking the maximum point of Z1 noise as sigma_1^2? This is power, it should be sqrt , no?? - Fixed it but think again
* Main.py Third Regime with 500 points is working with Gaussian, it had some infinite power issues when gd is active

--- TO DO ---
* Solve GD active Third Regime Main.py 
* Third Regime PP - Quantization implement
* Third Regime-Interference - Implement
    

## March 3rd
# NOTE: The maximum quantization location is chosen by only considering X alphabet, interference and noise is not taken 
# into consideration since the real mapping of X to Y matters. Make sure that this is a correct way to do it. 
# Consider updating this by minimizing the error between the real and quantized Y
* Third Regime PP- power was never divided with multiplying factor - Fixed
* Third Regime PP ADC is working
* Third Regime Interference is working

--- TO DO ---
* Focus on linear approximation