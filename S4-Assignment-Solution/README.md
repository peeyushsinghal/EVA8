Assignment
--------
Your new target is:
* 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
Do this in exactly 3 steps

Experiment 1
------------------------------------
**Objective / Target**

* Initial Setup and Model
* Getting the model correct
* Very basic model

**Results**

* Parameters: 6,379,786
* Best Train Accuracy: 99.91 %
* Best Test Accuracy: 99.28 %

**Analysis**

* Large but working model
* Overfitting (train - test accuracy) > 0

**Next Steps**

* Reduce Number of Params
* Remove overfitting


Experiment 2
--------------
**Objective / Target**
*   Reduce Params 
*   Reduce overfitting


**Results**

* Parameters: 4,038
* Best Train Accuracy: 97.73 %
* Best Test Accuracy: 98.23 %

**Analysis**

* Small model
* Not able to hit the accuracy mark
* Overfitting (train - test accuracy) < 0 contained

**Next Steps**

* Increase Number of Params
* little overfitting, not much

Experiment 3
----------------

**Objective / Target**
*   increase accuracy by increasing number of params
*   Induce some overfitting


**Results**

* Parameters: 9,790
* Best Train Accuracy: 98.88 %
* Best Test Accuracy: 99.13 %

**Analysis**

* Not hitting the accuracy mark
* Overfitting (train - test accuracy) < 0 is largely containted, model underfits (Target not achieved), However the train accuracy moves up and down, suggesting LR can be played with
* Number of params < 10K

**Next Steps**

* Increase accuracy
* Play with learning rate if it helps
