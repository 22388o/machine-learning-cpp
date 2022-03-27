/**
 * @file main.cpp
 * @author Sly Kint A. Bacalso
 * @date 2022-03-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include "./matplotlibcpp/matplotlibcpp.h"
#include "simplelinearregression.h"
using namespace std;

int main(){
    // variable X
    vector<double> X = { 8000, 6400, 2500, 3000, 6000, 5000, 8000, 4000, 11000, 25000, 4000, 8800, 5000, 7000, 8000, 1800, 5400, 15000, 3500, 2400, 1000, 8000, 2100, 8000, 4000, 1000, 2000, 4800
    };
    // variable Y
    vector<double> Y = { 38, 50, 15, 30, 50, 38, 50, 20, 45, 50, 20, 35, 30, 43, 35, 37.5, 37, 35, 30, 45, 4, 37.5, 25, 46, 30, 200, 200, 30,
    };
    
    
    double alpha = 0.0001; // learning rate
    int epoch = 1020;// number of epochs
    SimpleLinearRegression *slr = new SimpleLinearRegression(X, Y, alpha, epoch, true);
    slr->train();
    slr->print_yhat();

    
    vector<double> Y_c = slr->predict(X);
    // Normalize Y_c and X
    double X_MAX = *max_element(X.begin(), X.end());
    double X_MIN = *min_element(X.begin(), X.end()); 
    double Y_c_MAX = *max_element(Y_c.begin(), Y_c.end());
    double Y_c_MIN = *min_element(Y_c.begin(), Y_c.end());

    // Scatter plot
    matplotlibcpp::figure_size(700, 500);
    matplotlibcpp::scatter(Y, X, 25);

    double x = 5000;
    double y = slr->predict(x);

    cout << "Prediction of " << x << " Income is " << y << " Hours Per week" << endl;
    
    matplotlibcpp::plot({Y_c_MIN, Y_c_MAX}, {X_MIN, X_MAX}, "r");
    matplotlibcpp::xlabel("Hours per Week (x)");
    matplotlibcpp::ylabel("Income (y)");
    matplotlibcpp::title("Scatter Plot");
    matplotlibcpp::show();
};