# This code will read in the 16 positions and spit out the data needed for Luis's factorial design
import pandas as pd
import csv
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Gets the wpds and makes a regression tree model for each trial
def GetWpds(lpx,lpy, head_dir):   
    a = 0
    b = 0
    c = 0
    data_dir = head_dir+lpx+"/"+lpy+"/"
    file_path = data_dir+str(trial_list)+file_name
    wpd = pd.read_csv(file_path)  
    X = np.array(wpd.wsd_x.values)
    X = X.reshape(-1,1)
    y = np.array(wpd.mxpx_count.values)
    y = y.reshape(-1,1)
    regr = DecisionTreeRegressor(max_depth = 3) 
    regr.fit(X,y)   
    y_pred = regr.predict(X)    
    a = metrics.mean_absolute_error(y, y_pred)
    b = metrics.mean_squared_error(y, y_pred)
    c = np.sqrt(metrics.mean_squared_error(y, y_pred))
    d = r2_score(y,y_pred)   
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black",
            c="blue", label="data")
    plt.plot(X, y_pred, color="green",
         label="max_depth=3", linewidth=2)
    plt.xlabel("WSD")
    plt.ylabel("MXPX")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()     
    return a, b, c, d

# method for getting logistic curve bounds
def boundsForLogistic():
    trial = ["1/","2/","3/"]
    column_values = ["Actual", "Predicted", "WSD"]
    data_dir = head_dir+lpx+"/"+lpy+"/"
    wsdx_list = np.array([])
    mxpx_list = np.array([])
    wsd_min = np.array([])
    wsd_max = np.array([])
    for trial in trial:
        file_path = data_dir+trial+file_name
        wpd = pd.read_csv(file_path)        
        wsdx_list = np.concatenate((wsdx_list,wpd.wsd_x.values), axis = None)       
        mxpx_list = np.concatenate((mxpx_list,wpd.mxpx_count.values), axis = None)
        
    index = np.argsort(wsdx_list)
    X = wsdx_list[index].reshape(-1,1)
    y = mxpx_list[index].reshape(-1,1)
    regr = DecisionTreeRegressor(max_depth = 3) 
    test = regr.fit(X,y)
    y_pred = test.predict(X)
    
    shift_index = np.nonzero( np.diff(y_pred) )
    xp_shifts = X[shift_index].transpose()
    yp_shifts = y_pred[shift_index]
    
    # df = pd.DataFrame({"xp_shifts": xp_shifts, "yp_shifts": yp_shifts})
    # print(df)
    
    return xp_shifts, yp_shifts
    
# Creating final csv
def createCSVRegressionStats():
    i = 0
    with open('regressionTreeStats.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Row Position", "Column Position", "Trial Number", "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "Rsquared"])
            for x in LPX_LIST:
                for y in LPY_LIST:
                    for z in TRIAL_LIST:
                        writer.writerow([y,x,z, MAE[i], MSE[i], RMSE[i], Rsquared[i]])
                        i += 1
def createCSVRegressionTreeBounds():
    i = 0
    with open('regressionTreeBounds.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)  
            writer.writerow(["xp_shifts", "yp_shifts", "Row Position", "Column Position"])
            for x in LPX_LIST:
                for y in LPY_LIST:
                        writer.writerow([xp_shifts[i], yp_shifts[i], y,x])
                        i += 1            
            
                        
# Initializing some global variables 
file_name = "wpd.csv"
head_dir = "path" # Insert path
trial_list = ["1/","2/","3/"] 
LPY_LIST = ["00", "n5","01","p5"]
LPX_LIST = ["06","09","12","15"]
MAE = []
MSE = []
RMSE = []
Rsquared = []
TRIAL_LIST = [1,2,3]   
xp_shifts = []
yp_shifts = []                                      


# Calling Methods 
for x,lpx in enumerate(LPX_LIST):
    for y,lpy in enumerate(LPY_LIST):
        for trial in trial_list:
          z,x,v,m = GetWpds(lpx,lpy, head_dir) 
          MAE.append(z), MSE.append(x), RMSE.append(v), Rsquared.append(m)
          
for x,lpx in enumerate(LPX_LIST):
    for y,lpy in enumerate(LPY_LIST):
       q, w = boundsForLogistic()
       xp_shifts.append(q), yp_shifts.append(w)


createCSVRegressionTreeBounds() 
boundsForLogistic()               
createCSVRegressionStats()

