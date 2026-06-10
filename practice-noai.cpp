#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

/* 1. Turn the single labe into a vector for better comparison */
vector<float> iterateData(float& target, int& neurons) {
  vector<float> trg(neurons, 0.0F);
  if (static_cast<int>(target) >= 0 && static_cast<int>(target) < neurons) {
    trg[static_cast<int>(target)] = 1.0F;
  }
  return trg;
}

/* 2. Load the structured CSV File */
bool loadData(string& filename, vector<vector<float>>& i, vector<vector<float>>& a,
              int& neurons) {
  ifstream file(filename);
  if (!file.is_open()) {
    cerr << "[Error] File " << filename << " is not accessible or doesn't exist.\n";
    return false;
  }

  string line;
  int    count;
  while (getline(file, line)) {
    stringstream val(line);

    string chax;
    getline(val, chax, ',');
    float target = stof(chax);

    count++;
    vector<float> data;
    while (getline(val, chax, ',')) {
      data.push_back(stof(chax) / 255.0F);
    }
    i.push_back(std::move(data));
    a.push_back(std::move(iterateData(target, neurons)));

    if (count % 1000 == 0) {
      cout << "Loaded " << count << " Datasets\n";
    }
  }
  cout << "Completed Loading " << count << " Datasets.\n";
  return true;
}

/* 3. The ANN Object */
class Obj {
 private:
};