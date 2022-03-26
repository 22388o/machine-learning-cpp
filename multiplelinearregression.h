/**
 * @file multiplelinearregression.h
 * @author Sly Kint A. Bacalso
 * @date 2022-03-26
 * 
 * @copyright Copyright (c) 2022
 */

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
using namespace std;

class MultipleLinearRegression{
    double b0;
    vector<double> b; //handles b1 and b2
    
    double alpha;
    int epoch;
    bool normalize;
    vector<vector<double>> X;
    vector<double> Y;
    
    public:
        MultipleLinearRegression(vector<vector<double>> X, vector<double> Y, double alpha, int epoch, bool normalize = true) {
            this->X = X;
            this->Y = Y;
            this->alpha = alpha;
            this->epoch = epoch;
            b0 = 0.5;  // initial value of b0
            b = {0.8, 0.8}; // initial value of b1 and b2
            this->normalize = normalize;
        }

        void train() {
            // preparation for normalization
            vector<vector<double>> XY; 
            vector<double> X1;  
            vector<double> X2;  
            for(int i = 0; i < X.size(); i++){
                X1.push_back(X[i][0]); // X1 = X[:,0]
                X2.push_back(X[i][1]); // X2 = X[:,1]
            }

            if(normalize){
                // normalize X1 and X2 and the Y then concatenate them
                double X1_MAX = *max_element(X1.begin(), X1.end());
                double X1_MIN = *min_element(X1.begin(), X1.end());
                double X2_MAX = *max_element(X2.begin(), X2.end());
                double X2_MIN = *min_element(X2.begin(), X2.end());
                double Y_MAX = *max_element(Y.begin(), Y.end());
                double Y_MIN = *min_element(Y.begin(), Y.end());

                // normalization process and push them into XY
                for(int i = 0; i < X.size(); i++){
                    vector<double> temp = {
                        (X1[i] - X1_MIN)/(X1_MAX - X1_MIN), 
                        (X2[i] - X2_MIN)/(X2_MAX - X2_MIN),
                        (Y[i] - Y_MIN)/(Y_MAX - Y_MIN)
                    };
                    XY.push_back(temp);
                }
            } else {
                // no normalization just concatenate them
                for(int i = 0; i < X.size(); i++){
                    vector<double> temp = {X1[i], X2[i], Y[i]};
                    XY.push_back(temp);
                }
            }
            
            // begin the training
            for(int i = 1; i <= epoch; i++){
                // algorithm for random selection of data
                random_shuffle(XY.begin(), XY.end());
                
                // lets concat the X and Y
                vector<vector<double>> X1X2;
                vector<double> Y_new;
                for (int j = 0; j < XY.size(); j++) {
                    X1X2.push_back({XY[j][0], XY[j][1]});
                    Y_new.push_back(XY[j][2]);
                }
                vector<double> pred = predict(X1X2);
                double tot_err = loss(Y_new, pred); // MSE
                vector<double> D_b1 = dotmul2(subtract(Y_new, pred), X1X2); // gradient of b1
                double D_b0 = SumOfErrorDeviation(Y_new, pred);
                // update values of b0 and b1
                b = updateB(b, D_b1, alpha);
                b0 = b0 + alpha * D_b0;

                cout << "Epoch: " << i << " Total Error: " << tot_err << endl;

                if(tot_err < 0.001)
                    break;
            }
        }

        // gradient descent
        vector<double> updateB(vector<double> b, vector<double> D_b, double alpha) {
            vector<double> new_b;
            for(int i = 0; i < D_b.size(); i++){
                new_b.push_back(b[i] + alpha * D_b[i]);
            }
            return new_b;
        }

        // matrix subtraction
        vector<double> subtract(vector<double> y, vector<double> pred) {
            vector<double> result;
            for (int i = 0; i < y.size(); i++) {
                result.push_back(y[i] - pred[i]);
            }
            return result;
        }

        // Sum of error squared
        double SumOfErrorDeviation(vector<double> y, vector<double> pred) {
            double result = 0;
            for (int i = 0; i < y.size(); i++) {
                result += (y[i] - pred[i]) * 2;
            }
            return result;
        }

        // DOT product of 2 vectors
        vector<double> dotmul2(vector<double> a, vector<vector<double>> b) {
            double sum1 = 0, sum2 = 0;
            for(int i = 0; i < a.size(); i++){
                sum1 += a[i] * b[i][0];
                sum2 += a[i] * b[i][1];
            }
            return {sum1*2, sum2*2};
        }

        // Sum of mean square error
        double loss(vector<double> pred, vector<double> y) {
            double sum = 0;
            for (int i = 0; i < pred.size(); i++) {
                sum += pow(y[i] - pred[i], 2);
            }
            return sum / pred.size(); // MSE
        }

       
        void printYhat(){
            cout << endl << "Yhat = " << b0 << " + " << b[0] << "X1 + " << b[1] << "X2" << endl;
        }                

        // predict the value of Y
        double predict(double x1, double x2) {
            return b0 + b[0] * x1 + b[1] * x2;
        }
        
        // predict the value of Y for each x1 and x2
        vector<double> predict(vector<vector<double>> X) {
            vector<double> result;
            for (int i = 0; i < X.size(); i++) {
                result.push_back(predict(X[i][0], X[i][1]));
            }
            return result;
        }
};