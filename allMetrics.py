# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:33:19 2022

@author: Pierre-Eddy Dandrieux
"""

def allMetrics(model, X_test, y_test):
    # LIBRARIES
    import numpy as np
    from sklearn.metrics import confusion_matrix
    
    # COUNT CLASS
    _, numbClass = np.unique(y_test, return_counts=True)

    # PREDICTION          
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # METRIX FOLOWWING ONE PREDICTION
    tnPred, fpPred, fnPred, tpPred = confusion_matrix(y_test, y_pred).ravel()
    accPred = (tpPred + tnPred) / (tpPred + fnPred + fpPred + tnPred)
    recPred = tpPred / (tpPred + fnPred) 
    spePred = tnPred / (tnPred + fpPred)
    prePred = tpPred / (tpPred + fpPred)
    fprPred = 1 - spePred
    
    # METRICS FOR ROC-CURVE / PRECISION-RECALL-CURVE (INCREMENTAL P VALUE)
    # Number of sample / Variables pre-allocation
    nb_pts = 10000 
    prob = np.zeros([nb_pts, 1])
    tnProba = np.zeros([nb_pts, 1])
    fpProba = np.zeros([nb_pts, 1])
    fnProba = np.zeros([nb_pts, 1])
    tpProba = np.zeros([nb_pts, 1])
    accProba = np.zeros([nb_pts, 1])
    recProba = np.zeros([nb_pts, 1])
    speProba = np.zeros([nb_pts, 1])
    fprProba = np.zeros([nb_pts, 1])
    y_taff = np.c_[y_pred_proba, y_test]
    
    row = 0
    for pt in np.linspace(0, 1, nb_pts):
        # Filter of values with treshold of probability
        y_taff2 = y_taff[y_taff[:,1] >= pt]
        # Probabilities values at each iterations
        prob[row, 0] = pt 
        # True positive
        tpProba[row, 0] = np.sum(y_taff2[:,2], dtype='int') 
        # False negative                       
        fnProba[row, 0] = numbClass[1] - tpProba[row, 0] 
        # False positive
        fpProba[row, 0] = y_taff2.shape[0] - tpProba[row, 0]  
        # True negative               
        tnProba[row, 0] = numbClass[0] - fpProba[row, 0] 
        # Accuracy
        accProba[row, 0] = (tpProba[row, 0] + tnProba[row, 0]) / \
                                  (tpProba[row, 0] + fnProba[row, 0] + fpProba[row, 0] + tnProba[row, 0])
        # Recall, sensitivity, hit rate, true positive rate                          
        recProba[row, 0] = tpProba[row, 0] / (tpProba[row, 0] + fnProba[row, 0])
        # Specificity or true negative rate
        speProba[row, 0] = tnProba[row, 0] / (tnProba[row, 0] + fpProba[row, 0])
        # False positive rate or fall out
        fprProba[row, 0] = 1 - speProba[row, 0]
     
        row += 1
    del row, nb_pts, y_taff2
    
    # Roc auc score
    rocProba = np.absolute(np.trapz(recProba[:, 0], fprProba[:, 0]))
    
    # DICT TO SAVE
    metrics = {
        'proba' : prob,
        'tpProba' : tpProba,
        'fnProba' : fnProba,
        'fpProba' : fpProba,
        'tnProba' : tnProba,
        'accProba' : accProba,
        'recProba' : recProba,
        'fprProba' : fprProba,
        'speProba' : speProba,
        'rocProba' : rocProba,
        'tnPred' : tnPred,
        'fpPred' : fpPred, 
        'fnpred' : fnPred,
        'tpPred' : tpPred,
        'accPred' : accPred,
        'recPred' : recPred,
        'spePred' : spePred,
        'prePred' : prePred,
        'fprPred' : fprPred
        }
    
    return metrics
    del metrics
    