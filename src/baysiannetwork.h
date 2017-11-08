#ifndef BAYSIAN_NETWORK_H
#define BAYSIAN_NETWORK_H
#include "bitmap.hpp"
#include <vector>
#include <iostream>
#include <math.h>
#include <sstream>
#include <cmath>
#include <climits>
#include <fstream>
#include <iomanip>

using namespace std;

class BaysianNetwork {
private:
  vector<vector<unsigned char>> trainImages;
  vector<unsigned char> trainLabels;
  vector<vector<unsigned char>> testImages;
  vector<unsigned char> testLabels;
  vector<int> testClassifications;
  vector<double> classProbs;
  int numLabels;
  int numFeatures;
  vector <int> classCount;
  vector<vector<int>> numClassPredictions;
  //prob of F(j) = C(i)
  vector<vector<double>> classToPixelProb;

  //get probabilities of classes
  void getClassProbs();
  void getPixelConditionalProbs();
  void evaluate();
  void getClassPredictions();

public:
  BaysianNetwork(vector<vector<unsigned char>>& trImages, vector<unsigned char>& trLabels,
    vector<vector<unsigned char>>& ttImages, vector<unsigned char>& ttLabels, int numL, int numF);
  void outputBitmap();
  double percentAccuracy();
  void outputNetwork(string filename);
  void outputClassification(string filename);
};
#endif
