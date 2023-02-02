clear all
cd "STATA_FOLDER_PATH"

* importing data
import excel "STATA_QUERY.xlsx", firstrow

* dropping irrelevant rows
drop Unnamed0 Unnamed01

* renaming variables
rename HGoals HTGoals
rename AGoals ATGoals
rename HShotsTarget HTShotsTarget
rename AShotsTarget ATShotsTarget
rename HCorners HTCorners
rename ACorners ATCorners
rename HYellow HTYellow
rename AYellow ATYellow
rename HRed HTRed
rename ARed	ATRed
rename HGoalsAgainst HTGoalsAgainst
rename AGoalsAgainst ATGoalsAgainst
rename HShotsTargetAgainst HTShotsTargetAgainst
rename AShotsTargetAgainst ATShotsTargetAgainst

rename HGoals_5 HTGoals_5
rename AGoals_5 ATGoals_5
rename HShotsTarget_5 HTShotsTarget_5
rename AShotsTarget_5 ATShotsTarget_5
rename HCorners_5 HTCorners_5
rename ACorners_5 ATCorners_5
rename HYellow_5 HTYellow_5
rename AYellow_5 ATYellow_5
rename HRed_5 HTRed_5
rename ARed_5 ATRed_5
rename HGoalsAgainst_5 HTGoalsAgainst_5
rename AGoalsAgainst_5 ATGoalsAgainst_5
rename HShotsTargetAgainst_5 HTShotsTargetAgainst_5
rename AShotsTargetAgainst_5 ATShotsTargetAgainst_5

rename HNegSentimentDev HTNegSentimentDev
rename HNeuSentimentDev HTNeuSentimentDev
rename HPosSentimentDev HTPosSentimentDev
rename HCompoundSentimentDev HTCompoundSentimentDev
rename ANegSentimentDev ATNegSentimentDev
rename ANeuSentimentDev ATNeuSentimentDev
rename APosSentimentDev ATPosSentimentDev
rename ACompoundSentimentDev ATCompoundSentimentDev

* Creating outcome variable for the model. Home win = 1, draw = 2, away win = 3
generate probit_outcome = 1 if outcome == "1"
replace probit_outcome = 2 if outcome == "X"
replace probit_outcome = 3 if outcome == "2"

* exporting
export excel "data//STATA_QUERY_updated.xlsx", firstrow(variables) replace






/*
ORDERED PROBIT MODEL. THE MODELS CONSIDERED ARE

Model 1:
	- Only using rankings for both
	
Model 2:
	- Only using sentiment deviations for both

Model 3:
	- Only using attacking capabilites for both
	
Model 4:
	- Only using defensive probabilities for both
	
Model 5:
	- Using stepwise selection on pr(0.20), pe(0.10)
	
Model 6:
	- Using stepwise selection on pr(0.10), pe(0.05)
	
Model 7:	
	- Using stepwise selection on pr(0.05), pe(0.01)

Model 8:
	- Using all data
*/


/*
MODEL 1: ONLY USING RANKINGS FOR BOTH
*/

clear all
set more off

* Importing data
import excel "data//STATA_QUERY_updated.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTRank ATRank

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M1, title(Model 1)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') replace 

* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model1, pr outcome(1)
predict p2model1, pr outcome(2)
predict p3model1, pr outcome(3)

* Checking estimated probabilites
summarize p1model1 p2model1 p3model1
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model1 = .
replace prediction_model1 = 1 if p1model1 > p2model1 & p1model1 > p3model1
replace prediction_model1 = 2 if p2model1 > p1model1 & p2model1 > p3model1
replace prediction_model1 = 3 if p3model1 > p1model1 & p3model1 > p2model1

* Checking accuracy
generate correct_model1 = 1 if probit_outcome == prediction_model1
replace correct_model1 = 0 if probit_outcome != prediction_model1
summarize correct_model1 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace


/*
MODEL 2: ONLY USING SENTIMENT DEVIATIONS FOR BOTH
*/
clear all
set more off

* Importing data
import excel "data//STATA_QUERY_models.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTCompoundSentimentDev ATCompoundSentimentDev

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M2, title(Model 2)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') append 

* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model2, pr outcome(1)
predict p2model2, pr outcome(2)
predict p3model2, pr outcome(3)

* Checking estimated probabilites
summarize p1model2 p2model2 p3model2
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model2 = .
replace prediction_model2 = 1 if p1model2 > p2model2 & p1model2 > p3model2
replace prediction_model2 = 2 if p2model2 > p1model2 & p2model2 > p3model2
replace prediction_model2 = 3 if p3model2 > p1model2 & p3model2 > p2model2

* Checking accuracy
generate correct_model2 = 1 if probit_outcome == prediction_model2
replace correct_model2 = 0 if probit_outcome != prediction_model2
summarize correct_model2 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace

/*
MODEL 3: ONLY USING ATTACKING ATTRIBUTES FOR BOTH
*/
	
clear all
set more off

* Importing data
import excel "data//STATA_QUERY_models.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTGoals_5 HTShotsTarget_5 HTCorners_5 ATGoals_5 ATShotsTarget_5 ATCorners_5

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M3, title(Model 3)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') append 


* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model3, pr outcome(1)
predict p2model3, pr outcome(2)
predict p3model3, pr outcome(3)

* Checking estimated probabilites
summarize p1model3 p2model3 p3model3
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model3 = .
replace prediction_model3 = 1 if p1model3 > p2model3 & p1model3 > p3model3
replace prediction_model3 = 2 if p2model3 > p1model3 & p2model3 > p3model3
replace prediction_model3 = 3 if p3model3 > p1model3 & p3model3 > p2model3

* Checking accuracy
generate correct_model3 = 1 if probit_outcome == prediction_model3
replace correct_model3 = 0 if probit_outcome != prediction_model3
summarize correct_model3 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace

/*
MODEL 4: ONLY USING DEFENSIVE ATTRIBUTES FOR BOTH
*/
	
clear all
set more off

* Importing data
import excel "data//STATA_QUERY_models.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTGoalsAgainst_5 HTShotsTargetAgainst_5 HTYellow_5 HTRed_5 ATGoalsAgainst_5 ATShotsTargetAgainst_5 ATYellow_5 ATRed_5

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M4, title(Model 4)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') append 

* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model4, pr outcome(1)
predict p2model4, pr outcome(2)
predict p3model4, pr outcome(3)

* Checking estimated probabilites
summarize p1model4 p2model4 p3model4
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model4 = .
replace prediction_model4 = 1 if p1model4 > p2model4 & p1model4 > p3model4
replace prediction_model4 = 2 if p2model4 > p1model4 & p2model4 > p3model4
replace prediction_model4 = 3 if p3model4 > p1model4 & p3model4 > p2model4

* Checking accuracy
generate correct_model4 = 1 if probit_outcome == prediction_model4
replace correct_model4 = 0 if probit_outcome != prediction_model4
summarize correct_model4 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace

/*
MODEL 5: USING GETS-METHODS ON pr(0.20) and pe(0.10)
*/
	
clear all
set more off

* Importing data
import excel "data//STATA_QUERY_models.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTRank ATRank HTGoals_5 ATGoals_5 HTShotsTarget_5 ATShotsTarget_5 HTCorners_5 ATCorners_5 HTYellow_5 ATYellow_5 HTRed_5 ATRed_5 HTGoalsAgainst_5 ATGoalsAgainst_5 HTCompoundSentimentDev ATCompoundSentimentDev

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
stepwise, pr(0.20) pe(0.10): oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M5, title(Model 5)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') append 

* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model5, pr outcome(1)
predict p2model5, pr outcome(2)
predict p3model5, pr outcome(3)

* Checking estimated probabilites
summarize p1model5 p2model5 p3model5
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model5 = .
replace prediction_model5 = 1 if p1model5 > p2model5 & p1model5 > p3model5
replace prediction_model5 = 2 if p2model5 > p1model5 & p2model5 > p3model5
replace prediction_model5 = 3 if p3model5 > p1model5 & p3model5 > p2model5

* Checking accuracy
generate correct_model5 = 1 if probit_outcome == prediction_model5
replace correct_model5 = 0 if probit_outcome != prediction_model5
summarize correct_model5 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace
	
/*
MODEL 6: USING GETS-METHODS ON pr(0.10) and pe(0.05)
*/
	
clear all
set more off

* Importing data
import excel "data//STATA_QUERY_models.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTRank ATRank HTGoals_5 ATGoals_5 HTShotsTarget_5 ATShotsTarget_5 HTCorners_5 ATCorners_5 HTYellow_5 ATYellow_5 HTRed_5 ATRed_5 HTGoalsAgainst_5 ATGoalsAgainst_5 HTCompoundSentimentDev ATCompoundSentimentDev

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
stepwise, pr(0.10) pe(0.05): oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M6, title(Model 6)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') append 

* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model6, pr outcome(1)
predict p2model6, pr outcome(2)
predict p3model6, pr outcome(3)

* Checking estimated probabilites
summarize p1model6 p2model6 p3model6
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model6 = .
replace prediction_model6 = 1 if p1model6 > p2model6 & p1model6 > p3model6
replace prediction_model6 = 2 if p2model6 > p1model6 & p2model6 > p3model6
replace prediction_model6 = 3 if p3model6 > p1model6 & p3model6 > p2model6

* Checking accuracy
generate correct_model6 = 1 if probit_outcome == prediction_model6
replace correct_model6 = 0 if probit_outcome != prediction_model6
summarize correct_model6 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace
	

/*
MODEL 7: USING GETS-METHODS ON pr(0.05) and pe(0.01)
*/
	
clear all
set more off

* Importing data
import excel "data//STATA_QUERY_models.xlsx", firstrow

* Restricting dataset to same size as other models
drop if missing(HTGoals_5) | missing(HTShotsTarget_5) | missing(ATGoals_5) | missing(ATShotsTarget_5)

* Setting up dependent and independent variables
global ylist probit_outcome
global xlist HTRank ATRank HTGoals_5 ATGoals_5 HTShotsTarget_5 ATShotsTarget_5 HTCorners_5 ATCorners_5 HTYellow_5 ATYellow_5 HTRed_5 ATRed_5 HTGoalsAgainst_5 ATGoalsAgainst_5 HTCompoundSentimentDev ATCompoundSentimentDev

* Checking multicollinearity (model not used)
regress $ylist $xlist, vce(robust)
vif

* Ordered probit model coefficients (robust)
* Estimating model for 2021-2022 season (train data)
stepwise, pr(0.05) pe(0.01): oprobit $ylist $xlist if Season=="2021-2022", vce(robust)

* Accessing information criteria (AIC, BIC)
est store M7, title(Model 7)
estat ic
mat es_ic = r(S)
local AIC: display %4.1f es_ic[1,5]
local BIC: display %4.1f es_ic[1,6]

* Exporting model results
outreg2 using orderedprobit.xls, dec(3) addstat(AIC, `AIC', BIC, `BIC') append 

* Calculating marginal effects
mfx, predict(outcome(1))
mfx, predict(outcome(2))
mfx, predict(outcome(3))

* Predicted probabilites
predict p1model7, pr outcome(1)
predict p2model7, pr outcome(2)
predict p3model7, pr outcome(3)

* Checking estimated probabilites
summarize p1model7 p2model7 p3model7
tabulate probit_outcome

* Generating variable for prediction accuracy
generate prediction_model7 = .
replace prediction_model7 = 1 if p1model7 > p2model7 & p1model7 > p3model7
replace prediction_model7 = 2 if p2model7 > p1model7 & p2model7 > p3model7
replace prediction_model7 = 3 if p3model7 > p1model7 & p3model7 > p2model7

* Checking accuracy
generate correct_model7 = 1 if probit_outcome == prediction_model7
replace correct_model7 = 0 if probit_outcome != prediction_model7
summarize correct_model7 if Season == "2022-2023"

* Exporting results
export excel "data//STATA_QUERY_models.xlsx", firstrow(variables) replace



	
	
	
	
	
	
	
	
	
	
	
