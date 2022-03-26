/**
 * @file simplelinearregression.h
 * @author Sly Kint A. Bacalso
 * @date 2022-03-26
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <algorithm>
using namespace std;

class SimpleLinearRegression {
    double b0;
    double b1;
    double alpha;
    int epoch;
    bool normalize;
    vector<double> X;
    vector<double> Y;
    
    public:
        SimpleLinearRegression(vector<double> X, vector<double> Y, double alpha, int epoch, bool normalize = true) {
            this->X = X;
            this->Y = Y;
            this->alpha = alpha;
            this->epoch = epoch;
            b0 = 0;
            b1 = 0;
            this->normalize = normalize;
        }

        void train() {
            // preparation for data normalization
            vector<vector<double>> XY;
            if(normalize){
                double X_MAX = *max_element(X.begin(), X.end());
                double X_MIN = *min_element(X.begin(), X.end());
                double Y_MAX = *max_element(Y.begin(), Y.end());
                double Y_MIN = *min_element(Y.begin(), Y.end());
                // push to XY 2D vector with normalize data of X and Y
                for(int i = 0; i < X.size(); i++){
                    vector<double> temp = {
                        (X[i] - X_MIN)/(X_MAX - X_MIN), 
                        (Y[i] - Y_MIN)/(Y_MAX - Y_MIN)
                    };
                    XY.push_back(temp);
                }
            } else {
                for(int i = 0; i < X.size(); i++){
                    vector<double> temp = {X[i], Y[i]};
                    XY.push_back(temp);
                }
            }

            // begin training
            for(int i = 1; i <= epoch; i++){
                // algorithm for random selection of data
                random_shuffle(XY.begin(), XY.end());
                vector<double> X_new;
                vector<double> Y_new;
                // split the XY vector into X and Y
                for (int j = 0; j < XY.size(); j++) {
                    X_new.push_back(XY[j][0]);
                    Y_new.push_back(XY[j][1]);
                }
                // calculate the gradient
                vector<double> pred = predict(X_new);
                double tot_err = loss(Y_new, pred);
                double D_b1 = 2 * dot(subtract(Y_new, pred), X_new);
                double D_b0 = SumOfErrorDeviation(Y_new, pred);
                // update values of b0 and b1
                b1 = b1 + alpha * D_b1;
                b0 = b0 + alpha * D_b0;

                cout << "Epoch: " << i << " Total Error: " << tot_err << endl;

                if(tot_err < 0.01)
                    break;
            }
        }

        // matrix subtraction
        vector<double> subtract(vector<double> y, vector<double> pred) {
            vector<double> result;
            for (int i = 0; i < y.size(); i++) {
                result.push_back(y[i] - pred[i]);
            }
            return result;
        }

        // Sum of error deviation
        double SumOfErrorDeviation(vector<double> y, vector<double> pred) {
            double result = 0;
            for (int i = 0; i < y.size(); i++) {
                result += (y[i] - pred[i]) * 2;
            }
            return result;
        }

        // DOT product of two vectors
        double dot(vector<double> a, vector<double> b) {
            double sum = 0;
            for (int i = 0; i < a.size(); i++) {
                sum += a[i] * b[i];
            }
            return sum;
        }

        // Sum of mean squared error
        double loss(vector<double> pred, vector<double> y) {
            double sum = 0;
            for (int i = 0; i < pred.size(); i++) {
                sum += pow(y[i] - pred[i], 2);
            }
            return sum / pred.size(); // MSE
        }

        // predict the value of Y
        double predict(double x) {
            return b1 * x + b0;
        }

        void print_yhat(){
            cout << "Yhat: " << b0 << " + " << b1  << "x" << endl;
        }

        // predict the value of Y with X vector
        vector<double> predict(vector<double> x) {
            vector<double> pred;
            for(double data : x){
                pred.push_back(predict(data));
            }
            return pred;
        }
};