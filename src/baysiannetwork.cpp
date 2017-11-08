#include "baysiannetwork.h"


BaysianNetwork::BaysianNetwork(vector<vector<unsigned char>>& trImages,
  vector<unsigned char>& trLabels,vector<vector<unsigned char>>& ttImages,
  vector<unsigned char>& ttLabels, int numL, int numF){
    trainImages = trImages;
    trainLabels = trLabels;
    testImages = ttImages;
    testLabels = ttLabels;
    numLabels = numL;
    numFeatures = numF;
    getClassProbs();
    getPixelConditionalProbs();
    evaluate();
  }

void BaysianNetwork::getClassProbs(){

  for(int i = 0; i < numLabels; i++){
    classCount.push_back(0);
  }
  for(int i = 0; i < trainLabels.size(); i++){
    //get into of label
    int labelInt = static_cast<int>(trainLabels[i]);
    classCount[labelInt] ++;
  }
  //after summing, compute probs and put into classProbs
  //double sum = 0;
  for(int i = 0; i < classCount.size(); i++){
    double probForClass = (double)classCount[i] / (double)trainLabels.size();
    classProbs.push_back(probForClass);
    //sum += probForClass;
  }
  //check that it equals one
  //cout <<"Sum: " <<sum << endl;
}
void BaysianNetwork::getPixelConditionalProbs(){
  vector<vector<int>> classToPixelCount; //[class][feature]
  for(int i = 0; i < numLabels; i++){
    vector<int> pixelCount;
    classToPixelCount.push_back(pixelCount);
    for(int j = 0; j < numFeatures; j++){
      classToPixelCount[i].push_back(1); //using 1 instead of 0 because of the smoothing thing
    }
  }
  //seems inefficient
  //get the count of each feature = 1 per class
  for(int f = 0; f < numFeatures; f++){
    for(int c = 0; c < numLabels; c++){
      //go through each picture and count
      for(int trainPicIndex = 0; trainPicIndex < trainImages.size(); trainPicIndex++){
        if(static_cast<int>(trainLabels[trainPicIndex]) == c && static_cast<int>(trainImages[trainPicIndex][f]) == 1){
          classToPixelCount[c][f]++;
        }
      }
    }
  }
  //get the actuall probabilities using the Laplace smoothing
  for(int c = 0; c < numLabels; c++){
    vector<double> pixelProb;
    classToPixelProb.push_back(pixelProb);
    for(int f = 0; f < numFeatures; f++){
      //# of images of class c where pixel f is white
      double classPixel = (double)classToPixelCount[c][f];
      //number of images of class c + 2
      double cCount = (double)classCount[c] + 2;
      double conditionalProb = classPixel/cCount;
      classToPixelProb[c].push_back(conditionalProb);
      //cout << conditionalProb << endl;
    }
  }
}

void BaysianNetwork::evaluate(){
  for(int t = 0; t < testImages.size(); t++){
    double maxProb = INT_MIN;
    int maxClass = 0;
    for(int c = 0; c < numLabels; c++){
      double baySum = 0;
      for(int f = 0; f < numFeatures; f++){
        int testFeature = static_cast<int>(testImages[t][f]);
        double whiteTestClass = classToPixelProb[c][f];
        double testPixelClassProb;
        if(testFeature == 1){
          testPixelClassProb = whiteTestClass;
        }else{
          testPixelClassProb = 1 - whiteTestClass;
        }
        baySum += log(testPixelClassProb);
        //cout << " bay sum " << baySum << endl;
      }
      baySum += log(classProbs[c]);
      if(baySum > maxProb){
        maxProb = baySum;
        maxClass = c;
      }
    }
    testClassifications.push_back(maxClass);
  }
}
void BaysianNetwork::getClassPredictions(){
  for(int i = 0; i < 10; i ++){
    vector<int> vect;
    numClassPredictions.push_back(vect);
    for(int j = 0; j < 10; j++){
      numClassPredictions[i].push_back(0);
    }
  }
  for(int i = 0; i < testLabels.size(); i++){
    int actualLabel = static_cast<int>(testLabels[i]);
    int predictedLabel = testClassifications[i];
    numClassPredictions[actualLabel][predictedLabel]++;
  }
}


void BaysianNetwork::outputBitmap(){
  int numLabels = 10;
  int numFeatures = 784;
  for (int c=0; c<numLabels; c++) {
    std::vector<unsigned char> classFs(numFeatures);
    for (int f=0; f<numFeatures; f++) {
        double p = classToPixelProb[c][f];
        uint8_t v = 255*p;
        classFs[f] = (unsigned char)v;
      }
      std::stringstream ss;
      ss << "../output/digit" <<c<<".bmp";
      Bitmap::writeBitmap(classFs, 28, 28, ss.str(), false);
    }
}

double BaysianNetwork::percentAccuracy(){
  int numRight = 0;
  for(int i = 0; i < testLabels.size(); i ++){
    if(static_cast<int>(testLabels[i]) == testClassifications[i]){
      numRight ++;
    }
  }
  return (double)numRight/(double)testLabels.size();
}

void BaysianNetwork::outputNetwork(string filename){
  ofstream myfile;
  myfile.open (filename);
  //prob of each feature for each class
  for(int c = 0; c < numLabels; c++){
    for(int f = 0; f < numFeatures; f++){
      myfile << classToPixelProb[c][f] << endl;
    }
  }
  //prob of each class
  for(int c = 0; c < numLabels; c++){
    myfile << classProbs[c] << endl;
  }
  myfile.close();
}

void BaysianNetwork::outputClassification(string filename){
  getClassPredictions();
  ofstream myfile;
  myfile.open (filename);
  //prob of each feature for each class
  for(int i = 0; i < 10; i++){
    for(int j = 0; j < 10; j++){
      myfile << setw(6) << numClassPredictions[i][j];
    }
    myfile << endl;
  }
  //accuracy
  myfile << percentAccuracy();
  myfile.close();
}
