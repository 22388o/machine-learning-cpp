/**
 * @file main2.cpp
 * @author Sly Kint A. Bacalso
 * @date 2022-03-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include "multiplelinearregression.h"
using namespace std;

int main(){
    // 2 independent variables X1 and X2
    vector<vector<double>> X ={
        {1, 0},
        {1, 1},
        {0, 0},
        {0, 1}
    };
    // dependent variable Y
    vector<double> Y = {0, 1, 0, 0};
    
    
    double alpha = 0.0001; // learning rate
    int epoch = 1000; // number of epochs
    
    MultipleLinearRegression *mlr = new MultipleLinearRegression(X, Y, alpha, epoch, false);
    mlr->train();

    // Start AND gate prediction

    // Test 1 prediction of 1 AND 1 should be 1
    int x1 = 1; int x2 = 1;
    int y = mlr->predict(x1, x2);
    mlr->printYhat();
    cout << "y = " << mlr->predict(x1, x2) << endl;
    cout << "AND Gate Prediction:" << endl;
    cout << "Prediction of " << x1 << " AND " << x2 << " is " << y << endl;

    // Test 2 Prediction of 1 AND 0 should be 0
    x1 = 1; x2 = 0;
    y = mlr->predict(x1, x2);
    mlr->printYhat();
    cout << "y = " << mlr->predict(x1, x2) << endl;
    cout << "AND Gate Prediction:" << endl;
    cout << "Prediction of " << x1 << " AND " << x2 << " is " << y << endl;

    // Test 3 Prediction of 0 AND 1 should be 0
    x1 = 0; x2 = 1;
    y = mlr->predict(x1, x2);
    mlr->printYhat();
    cout << "y = " << mlr->predict(x1, x2) << endl;
    cout << "AND Gate Prediction:" << endl;
    cout << "Prediction of " << x1 << " AND " << x2 << " is " << y << endl;

    // Test 4 Prediction of 0 AND 0 should be 0
    x1 = 0; x2 = 0;
    y = mlr->predict(x1, x2);
    mlr->printYhat();
    cout << "y = " << mlr->predict(x1, x2) << endl;
    cout << "AND Gate Prediction:" << endl;
    cout << "Prediction of " << x1 << " AND " << x2 << " is " << y << endl;
};