#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

vector<float> createOneHot(int label, int classes) {
  vector<float> one_hot(classes, 0.0F);
  if (label >= 0 && label < classes) {
    one_hot[label] = 1.0F;
  }
  return one_hot;
}

bool loadCSV(const string &filename, vector<vector<float>> &inputs,
             vector<vector<float>> &actuals, int classes) {
  ifstream file(filename);
  if (!file.is_open()) {
    cerr << "[ERROR]: Could not open file " << filename << "\n";
    return false;
  }

  string line;
  int count = 0;

  while (getline(file, line)) {
    string value;
    stringstream ss(line);
    getline(ss, value, ',');
    int label = stoi(value);

    vector<float> pixels;
    pixels.reserve(784);

    while (getline(ss, value, ',')) {
      pixels.push_back(stof(value) / 255.0F);
    }

    inputs.push_back(std::move(pixels));
    actuals.push_back(createOneHot(label, classes));

    ++count;
    if (count % 1000 == 0) {
      cout << "Loaded " << count << " Samples\n";
    }
  }
  cout << "Loaded Total " << count << " Samples\n";
  return true;
}

class ANN {
private:
  vector<vector<vector<float>>> weights;
  vector<vector<float>> biases;
  vector<vector<float>> last_z;
  vector<vector<float>> last_a;
  float learning_rate;

  vector<vector<float>> initMatrix(int rows, int cols) {
    vector<vector<float>> mtrx(rows, vector<float>(cols, 0.0F));

    random_device rd;
    mt19937 gen(rd());
    float stddev = sqrt(2.0F / static_cast<float>(cols));
    normal_distribution<float> dis(0, stddev);

    for (auto &row : mtrx) {
      for (auto &col : row) {
        col = dis(gen);
      }
    }
    return mtrx;
  }

  vector<float> initBiasVctr(int size) { return vector<float>(size, 0.0F); }

  vector<float> mtrxMul(vector<vector<float>> &mtrx, vector<float> &bias) {
    int rows = static_cast<int>(mtrx.size());
    int cols = static_cast<int>(mtrx[0].size());
    vector<float> result(rows, 0.0F);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[i] += mtrx[i][j] * bias[j];
      }
    }

    return result;
  }

  vector<float> applyReLU(vector<float> &z) {
    vector<float> a(z.size());
    for (size_t i = 0; i < z.size(); i++) {
      a[i] = (z[i] > 0) ? z[i] : (0.01 * z[i]);
    }

    return a;
  }

  vector<float> deriveReLU(vector<float> &leakyReLU) {
    vector<float> a(leakyReLU.size());

    for (size_t i = 0; i < leakyReLU.size(); i++) {
      a[i] = (leakyReLU[i] > 0) ? 1.0F : 0.1F;
    }
    return a;
  }

public:
  ANN(vector<float> &topology, float lr) : learning_rate(lr) {
    int num_layers = static_cast<int>(topology.size()) - 1;
    weights.resize(num_layers);
    biases.resize(num_layers);
    last_z.resize(num_layers);
    last_a.resize(num_layers + 1);

    for (int i = 0; i < num_layers; i++) {
      weights[i] = initMatrix(topology[i + 1], topology[i]);
      biases[i] = initBiasVctr(topology[i + 1]);
    }
  }

  vector<float> predict(vector<float> &input) {
    last_a[0] = input;

    int num_layers = static_cast<int>(weights.size());
    for (int i = 0; i < num_layers; i++) {
      last_z[i] = mtrxMul(weights[i], last_a[i]);
      for (int l = 0; l < static_cast<int>(last_z.size()); i++) {
        last_z[i][l] += biases[i][l];
      }
      last_a[i + 1] = applyReLU(last_z[i]);
    }
    return last_a[num_layers];
  }

  float calcMSE(vector<float> &predicted, vector<float> &actual) {
    float err_total = 0.0F;
    for (size_t i = 0; i < predicted.size(); i++) {
      float err = predicted[i] - actual[i];
      err_total += err * err;
    }
    return err_total / static_cast<float>(predicted.size());
  }

  void backPropagate(const vector<float> &actual) {
    int num_layers = static_cast<int>(weights.size());
    const vector<float> &predicted = last_a[num_layers];

    vector<vector<float>> deltas(num_layers);
    int l = num_layers - 1;
    vector<float> rd = deriveReLU(last_z[l]);
    deltas[l].resize(predicted.size());

    for (size_t i = 0; i < predicted.size(); i++) {
      deltas[l][i] = (predicted[i] - actual[i]) * rd[i];
    }

    for (int z = num_layers - 2; z >= 0; z--) {
      rd = deriveReLU(last_z[z]);
      int size_z = static_cast<int>(weights[z].size());
      int size_z_next = static_cast<int>(weights[z + 1].size());

      deltas[z].assign(size_z, 0.0F);

      for (int i = 0; i < size_z; i++) {
        for (int k = 0; k < size_z_next; k++) {
          deltas[z][i] += weights[z + 1][k][i] * deltas[z + 1][k];
        }
      }

      for (int i = 0; i < size_z; i++) {
        deltas[z][i] += rd[i];
      }
    }

    for (int l = 0; l < num_layers; l++) {
      int rows = static_cast<int>(weights[l].size());
      int cols = static_cast<int>(weights[l][0].size());

      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          weights[l][i][j] -= learning_rate * deltas[l][i] * last_a[l][j];
        }
        biases[l][i] -= learning_rate * deltas[l][i];
      }
    }
  }
};