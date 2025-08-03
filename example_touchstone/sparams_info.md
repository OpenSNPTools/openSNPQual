# Quality Metrics Calculated for The Example Touchstone Files

## Change log
**2025-08-02:** added correlation to IEEE370 Time Domain metric CAUSALITY including the Causality matrix, status: OK. Stripline S-parameters correlate. 

**2025-07-17:** added correlation to IEE370 Freq domain all metrics; status: OK. All 4 sample s-parameters correlate.

## pcb_stripline_119mm.s2p

Frequency domain result comparison:

| Method FREQ | Causality         | Passivity         | Reciprocity       |
|-------------|-------------------|-------------------|-------------------|
| IEEE370     | 9.7139            | 99.9999           | 94.1444           |
| OpenSNPQual | 9.713891023717428 | 99.99993110422362 | 94.14443314741024 |

---

Time domain calculation settings:

	data_rate = 25.125; #data rate in Gbps
	rise_per = 0.4; # rise time - fraction of UI
	sample_per_UI = 32;
	pulse_shape = 1; #1 is Gaussian; 2 is Rectangular with Butterworth filter; 3 is Rectangular with Gaussian filter;
	extrapolation_method = 2; #1 is constant extrapolation; 2 is zero padding;
	

| Method TIME | Causality         | Passivity         | Reciprocity       |
|-------------|-------------------|-------------------|-------------------|
| IEEE370     | 6.655000          | 0.010000          | 3.380000          |
| OpenSNPQual | 6.655             | 0.015             | 3.465             |
   


#### Matrix: causality_time_domain_difference_mv

| IEEE370 | 1           | 2           |
|---------|-------------|-------------|
| 1       | 9.5960e-03  | 5.7716e-03  |
| 2       | 5.2785e-03  | 5.0836e-03  |
| OpenSNPQual | 1             | 2             |
| 1           | 9.593728e-03  | 5.771632e-03  |
| 2           | 5.279385e-03  | 5.083631e-03  |


reciprocity_time_domain_difference_mv =

   0   6.7615e-03
   
   6.7615e-03            0

passivity_time_domain_difference_mv =

   8.8767e-06   1.3961e-05
   
   1.3241e-05   8.8938e-06


## pcb_stripline_238mm.s2p

Frequency domain result comparison:

| Method Freq | Causality         | Passivity         | Reciprocity       |
|-------------|-------------------|-------------------|-------------------|
| IEEE370     | 11.512            | 100.000           | 96.853            |
| OpenSNPQual | 11.51155849954462 | 99.9994453028125  | 96.85330004075794 |


---

| Method TIME | Causality         | Passivity         | Reciprocity       |
|-------------|-------------------|-------------------|-------------------|
| IEEE370     | 5.6000            | 0                 | 2.4500            |
| OpenSNPQual | 5.595             | 0.015             | 2.43              |

  
#### Matrix: causality_time_domain_difference_mv

| IEEE370 | 1           | 2           |
|---------|-------------|-------------|
| 1       | 7.7848e-03   | 5.0713e-03  |
| 2       | 4.2552e-03   | 4.7415e-03  |
| OpenSNPQual | 1             | 2             |
| 1           | 7.784144e-03  | 5.071260e-03  |
| 2           | 4.266165e-03  | 4.741466e-03  |


---
     

reciprocity_time_domain_difference_mv =

   0   4.8609e-03
   
   4.8609e-03            0

passivity_time_domain_difference_mv =

   7.1764e-06   1.3544e-05
   
   1.3108e-05   7.1819e-06


## CABLE1_TX_pair.s4p

Frequency domain result comparison:

| Method Freq | Causality         | Passivity         | Reciprocity       |
|-------------|-------------------|-------------------|-------------------|
| IEEE370     | 96.435            | 100.000           | 99.238            |
| OpenSNPQual | 96.43533639099226 | 100.0             | 99.23771428457677 |


## CABLE1_RX_pair.s4p

Frequency domain result comparison:

| Method Freq | Causality         | Passivity         | Reciprocity       |
|-------------|-------------------|-------------------|-------------------|
| IEEE370     | 97.717            | 100.000           | 99.188            |
| OpenSNPQual | 97.7173628796294  | 100.0             | 99.18805318009613 |

